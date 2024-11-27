import os
import re
import sys
import glob
import json
import torch
import datetime
import math
from typing import Tuple
import itertools

from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime
from time import sleep
from tqdm import tqdm

#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from ranger import Ranger

from accelerate import Accelerator

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
#import rvc.lib.zluda

from infer.lib.train.utils import (
    HParams,
    plot_spectrogram_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
    get_hparams,
)

from infer.lib.train.data_utils import DistributedBucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid

from infer.lib.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss

from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, MultiScaleMelSpectrogramLoss
from infer.lib.train.extract_model import extract_model

from rvc.layers.synthesizers import Synthesizer
from rvc.layers.algorithm import commons
from rvc.layers.utils import slice_on_last_dim, total_grad_norm

from rvc.layers.discriminators.sub.mpd import MultiPeriodDiscriminatorV2
from rvc.layers.discriminators.sub.mssbcqtd import MultiScaleSubbandCQTDiscriminator
from rvc.layers.discriminators.sub.msstftd import MultiScaleSTFTDiscriminator

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

# Torch backends config
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# Parse command line arguments
hps = get_hparams()
model_name = hps.name
save_every_epoch = hps.save_every_epoch
total_epoch = hps.total_epoch
pretrainG = hps.pretrainG
pretrainD = hps.pretrainD
version = hps.version
gpus = hps.gpus
batch_size = hps.train.batch_size
sample_rate = hps.data.sampling_rate
pitch_guidance = hps.if_f0
save_only_latest = hps.if_latest
save_every_weights = hps.save_every_weights
cache_data_in_gpu = hps.if_cache_data_in_gpu
use_spectral_norm = hps.model.use_spectral_norm
current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "0_gt_wavs")


with open(config_save_path, "r") as f:
    config = json.load(f)
    
config = HParams(**config)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")


# Globals
global_step = 0
warmup_epochs = 5
use_warmup = False
warmup_completed = False


# --------------------------   Execution   --------------------------
def main():
    """
    Main function to start the training process.
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))


    run(experiment_dir,
        pretrainG,
        pretrainD,
        pitch_guidance,
        total_epoch,
        save_every_weights,
        config,
    )


# --------------------------   Setup / Initialization   --------------------------
def initialize_accelerator(config):
    # Conditioned enabling for mixed precision training based on config's "fp16_run"
    if config.train.fp16_run:
        accelerator = Accelerator(mixed_precision="fp16")  # If you wanna use brainfloat16 precision instead, replace "fp16" with "bf16".
    else:
        accelerator = Accelerator(mixed_precision="no")  # If fp16_run is set to false, training is done in fp32

    return accelerator


# --------------------------   Data Handling / Loading   --------------------------
def prepare_data_loaders(config, batch_size, n_gpus, rank):

    # Creates datasets and dataloaders
    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()

    # Groups samples into length-based buckets to minimize padding. Ensures an even distribution of samples across multiple workers.
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,  # Total batch_size across all GPUs.    ( batch_size x n_gpus = Global batch size.   ex.: batch_size 2 [LOCAL] * 8 = batch_size 16 [GLOBAL] ) 
        [100, 200, 300, 400, 500, 600, 700, 800, 900],   # Bucket sizes for the sampler
        num_replicas=n_gpus,  # Number of replicas (GPUs)
        rank=rank,
        shuffle=True, 
    )
    print("batch_size global", {batch_size})
    print(f"number of replicas: {n_gpus}")
    # Initializes the train_loader
    train_loader = DataLoader(
        train_dataset,  # Dataset to load
        num_workers=4,  # Number of CPU threads used for data loading
        shuffle=False,  # No need for shuffle in here as it's already managed by "train_sampler"
        pin_memory=True,  # Pin memory for faster GPU transfer
        collate_fn=collate_fn,  # "TextAudioCollateMultiNSFsid" Collator
        batch_sampler=train_sampler,  # Uses the DistributedBucketSampler as batch_sampler
        persistent_workers=True,  # Keeps workers alive between epochs
        prefetch_factor=8,  # How many batches to prefetch for each worker.  Default is: 8          ////////  ADJUST IF YOU EXPERIENCE OOM - Cuda Out of memory  error ////////
    )
    return train_loader


# --------------------------   Custom functions land in here   --------------------------


# Mel spectrogram similarity metric ( Predicted ∆ Real ) using L1 loss
def mel_spectrogram_similarity(y_hat_mel, y_mel):
    # Calculate the L1 loss between the generated mel and original mel spectrograms
    loss_mel = F.l1_loss(y_hat_mel, y_mel)

    # Convert the L1 loss to a similarity score between 0 and 100
    # Scale the loss to a percentage, where a perfect match (0 loss) gives 100% similarity
    mel_spec_similarity = 100.0 - (loss_mel.item() * 100.0)

    # Convert the similarity percentage to a tensor
    mel_spec_similarity = torch.tensor(mel_spec_similarity)

    # Clip the similarity percentage to ensure it stays within the desired range
    mel_spec_similarity = torch.clamp(mel_spec_similarity, min=0.0, max=100.0)

    return mel_spec_similarity


# Helper function for conditional ( based on n_gpus ) gathered / non gathered loss returning
def process_gathered_loss(name, loss, accelerator):
    if accelerator.num_processes > 1:
        # Gather loss and compute the mean across GPUs
        gathered_loss = accelerator.gather(loss)
        loss_to_log = gathered_loss.mean().item()
    else:
        # Use the original loss directly
        loss_to_log = loss.item()

    # Return the processed loss to be logged later, not logging here
    return loss_to_log


# --------------------------   Custom functions End here   --------------------------


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"


def initialize_models_and_optimizers(config, version, pitch_guidance, device):

    # Initialize Generator:
    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=pitch_guidance == True,  # converting 1/0 to True/False
        is_half=Accelerator.mixed_precision == "fp16",
        sr=sample_rate,
    )

    # Initialize Discriminators:
    net_d_mpd = MultiPeriodDiscriminatorV2(use_spectral_norm)
    net_d_cqt = MultiScaleSubbandCQTDiscriminator(filters=128, sample_rate=sample_rate)
    net_d_stft = MultiScaleSTFTDiscriminator(filters=128)

    # Define the optimizers
    optim_g = Ranger(
        net_g.parameters(),
        lr = 0.0001,
        betas = (0.8, 0.99), # 0.9, 0.999
        eps = 1e-8,
        weight_decay = 0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5, # 4 or 5 can be tried
        use_gc=False,
        gc_conv_only=False,
        gc_loc=False,
    )
    optim_d_mpd = Ranger(
        net_d_mpd.parameters(),
        lr = 1e-4,
        betas = (0.8, 0.99), # 0.9, 0.999
        eps = 1e-8,
        weight_decay = 0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5, # 4 or 5 can be tried
        use_gc=False,
        gc_conv_only=False,
        gc_loc=False,
    )
    optim_d_cqt = Ranger(
        net_d_cqt.parameters(),
        lr = 1e-4, # Worth trying: 4e-4 and especially 5e-4 or 9e-5
        betas = (0.8, 0.99), # 0.9, 0.999
        eps = 1e-8,
        weight_decay = 0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5, # 4 or 5 can be tried
        use_gc=False,
        gc_conv_only=False,
        gc_loc=False,
    )
    optim_d_stft = Ranger(
        net_d_stft.parameters(),
        lr = 1e-4, # Worth trying: 5e-4
        betas = (0.8, 0.99), # 0.9, 0.999
        eps = 1e-8,
        weight_decay = 0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5, # 4 or 5 can be tried
        use_gc=False,
        gc_conv_only=False,
        gc_loc=False,
    )
    return net_g, net_d_mpd, net_d_cqt, net_d_stft, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft


def load_model_checkpoint(
    hps,
    train_len,
    net_g,
    net_d_mpd,
    net_d_cqt,
    net_d_stft,
    optim_g,
    optim_d_mpd,
    optim_d_cqt,
    optim_d_stft,
    rank,
    accelerator
    ):

    # Load checkpoint if available
    try:
        print("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"),
            [accelerator.unwrap_model(net_d_mpd), accelerator.unwrap_model(net_d_cqt), accelerator.unwrap_model(net_d_stft)],
            [optim_d_mpd, optim_d_cqt, optim_d_stft]
        )

        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"),
            [accelerator.unwrap_model(net_g)],
            [optim_g]
        )

        epoch_str += 1
        global_step = (epoch_str - 1) * train_len


    except:
        epoch_str = 1
        global_step = 0

        # Loading the pretrained Generator model
        if pretrainG != "":
            if rank == 0:
                print(f"Loading pretrained (G) '{pretrainG}'")
            # Load the generator model
            checkpoint = torch.load(pretrainG, map_location="cpu")
            state_dicts = checkpoint["models"]
            accelerator.unwrap_model(net_g).load_state_dict(state_dicts["model_1"])

        # Loading the pretrained Discriminator models
        if pretrainD != "":
            if rank == 0:
                print(f"Loading pretrained (D) '{pretrainD}'")
            # Load the discriminator models
            checkpoint = torch.load(pretrainD, map_location="cpu")
            state_dicts = checkpoint["models"]

            # MultiPeriodDiscriminator:
            accelerator.unwrap_model(net_d_mpd).load_state_dict(state_dicts["model_1"])
            # MultiScale SubBand CQT Discriminator:
            accelerator.unwrap_model(net_d_cqt).load_state_dict(state_dicts["model_2"]) 
            # MultiScale STFT Discriminator:
            accelerator.unwrap_model(net_d_stft).load_state_dict(state_dicts["model_3"])

    return epoch_str, global_step


def setup_schedulers(config, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft, epoch_str, warmup_epochs, use_warmup):

    # Initialize warmup schedulers to None in case they are not needed
    warmup_scheduler_g = None
    warmup_scheduler_d_mpd = None
    warmup_scheduler_d_cqt = None
    warmup_scheduler_d_stft = None

    if use_warmup:
        # Warmup for: Generator
        warmup_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            optim_g,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        )
        # Warmup for: MPD
        warmup_scheduler_d_mpd = torch.optim.lr_scheduler.LambdaLR(
            optim_d_mpd,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        )

        # Warmup for: CQT
        warmup_scheduler_d_cqt = torch.optim.lr_scheduler.LambdaLR(
            optim_d_cqt,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        )

        # Warmup for: STFT
        warmup_scheduler_d_stft = torch.optim.lr_scheduler.LambdaLR(
            optim_d_stft,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        )

    # Ensure initial_lr is set when use_warmup is False
    if not use_warmup:
        # For: Generator
        for param_group in optim_g.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        # For: MPD
        for param_group in optim_d_mpd.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        # For: CQT
        for param_group in optim_d_cqt.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        # For: STFT
        for param_group in optim_d_stft.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']


    # For the decay phase (after warmup)
    decay_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 1
    )
    decay_scheduler_d_mpd = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_mpd, gamma=config.train.lr_decay, last_epoch=epoch_str - 1
    )

    decay_scheduler_d_cqt = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_cqt, gamma=config.train.lr_decay, last_epoch=epoch_str - 1
    )
    decay_scheduler_d_stft = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_stft, gamma=config.train.lr_decay, last_epoch=epoch_str - 1
    )

    return warmup_scheduler_g, warmup_scheduler_d_mpd, warmup_scheduler_d_cqt, warmup_scheduler_d_stft, decay_scheduler_g, decay_scheduler_d_mpd, decay_scheduler_d_cqt, decay_scheduler_d_stft


def run(
    experiment_dir,
    pretrainG,
    pretrainD,
    pitch_guidance,
    custom_total_epoch,
    custom_save_every_weights,
    config,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        pitch_guidance (bool): Flag indicating whether to use pitch guidance during training.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_epochs, use_warmup, warmup_completed


    accelerator = initialize_accelerator(config)
    n_gpus = accelerator.num_processes
    rank = accelerator.process_index
    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=experiment_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))

    torch.manual_seed(config.train.seed)

    train_loader = prepare_data_loaders(config, batch_size, n_gpus, rank)
    net_g, net_d_mpd, net_d_cqt, net_d_stft, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft = initialize_models_and_optimizers(config, version, pitch_guidance, accelerator.device)
    epoch_str, global_step = load_model_checkpoint(config, len(train_loader), net_g, net_d_mpd, net_d_cqt, net_d_stft, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft, rank, accelerator)
    warmup_scheduler_g, warmup_scheduler_d_mpd, warmup_scheduler_d_cqt, warmup_scheduler_d_stft, decay_scheduler_g, decay_scheduler_d_mpd, decay_scheduler_d_cqt, decay_scheduler_d_stft = setup_schedulers(config, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft, epoch_str, warmup_epochs, use_warmup)

    # wrapping
    train_loader, net_g, net_d_mpd, net_d_cqt, net_d_stft, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft, warmup_scheduler_g, warmup_scheduler_d_mpd, warmup_scheduler_d_cqt, warmup_scheduler_d_stft, decay_scheduler_g, decay_scheduler_d_mpd, decay_scheduler_d_cqt, decay_scheduler_d_stft = accelerator.prepare(train_loader, net_g, net_d_mpd, net_d_cqt, net_d_stft, optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft, warmup_scheduler_g, warmup_scheduler_d_mpd, warmup_scheduler_d_cqt, warmup_scheduler_d_stft, decay_scheduler_g, decay_scheduler_d_mpd, decay_scheduler_d_cqt, decay_scheduler_d_stft)

    cache = []
   # get the first sample as reference for tensorboard evaluation
    for info in train_loader:
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        reference = (
            phone.to(accelerator.device),
            phone_lengths.to(accelerator.device),
            pitch.to(accelerator.device) if pitch_guidance else None,
            pitchf.to(accelerator.device) if pitch_guidance else None,
            sid.to(accelerator.device),
        )
        break

    for epoch in range(epoch_str, custom_total_epoch + 1):
        train_and_evaluate(
            epoch,
            config,
            [net_g, net_d_mpd, net_d_cqt, net_d_stft],
            [optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft],
            #scaler,
            train_loader,
            writer if rank == 0 else None,
            cache,
            custom_save_every_weights,
            custom_total_epoch,
            accelerator,
            reference
        )
        if use_warmup and epoch <= warmup_epochs:
            # Starts the warmup phase if warmup_epochs =/= warmup_epochs
            warmup_scheduler_g.step()
            warmup_scheduler_d_mpd.step()
            warmup_scheduler_d_cqt.step()
            warmup_scheduler_d_stft.step()

            # Logging of finished warmup
            if epoch == warmup_epochs:
                warmup_completed = True
                print(f"//////  Warmup completed at warmup epochs:{warmup_epochs}  //////")
                # Gen:
                print(f"//////  LR G: {optim_g.param_groups[0]['lr']}  //////")
                # Discs:
                print(f"//////  LR D_MPD: {optim_d_mpd.param_groups[0]['lr']}  //////")
                print(f"//////  LR D_CQT: {optim_d_cqt.param_groups[0]['lr']}  //////")
                print(f"//////  LR D_STFT: {optim_d_stft.param_groups[0]['lr']}  //////")
                # Decay gamma:
                print(f"//////  Starting the exponential lr decay with gamma of {config.train.lr_decay}  //////")
 
        # Once the warmup phase is completed, uses exponential lr decay
        if not use_warmup or warmup_completed:
            decay_scheduler_g.step()
            decay_scheduler_d_mpd.step()
            decay_scheduler_d_cqt.step()
            decay_scheduler_d_stft.step()


def train_and_evaluate(
    epoch,
    hps,
    nets,
    optims,
    #scaler,
    train_loader,
    writer,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    accelerator,
    reference
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        scaler (GradScaler): Gradient scaler for mixed precision training.
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer, writer_eval].
        cache (list): List to cache data in GPU memory.
    """
    global global_step

    # Initialize the models and optimizers
    net_g, net_d_mpd, net_d_cqt, net_d_stft = nets
    optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft = optims

    train_loader.batch_sampler.set_epoch(epoch)
    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)

    net_g.train()
    net_d_mpd.train()
    net_d_cqt.train()
    net_d_stft.train()

    device = accelerator.device
    rank = accelerator.process_index

    # Conditioned data caching on GPU
    if cache_data_in_gpu: # If data caching is enabled
        data_iterator = cache
        if not cache: # If cache is empty, populate with data
            for batch_idx, info in enumerate(train_loader):
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
                # Whether model uses pitch guidance ( f0 ) or not
                pitch = pitch if pitch_guidance else None
                pitchf = pitchf if pitch_guidance else None
                cache.append(
                    (batch_idx, (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid))
                )
        else: # If cache is not empty, shufle data
            shuffle(cache)
    else: # If data caching on GPU is not enabled:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    # Over N mini-batches loss averaging
    N = 90  # Number of mini-batches after which the loss is logged
    running_loss_gen = 0.0  # Running loss for generator
    running_loss_disc = 0.0  # Running loss for discriminator


    # Main training loop with progress bar
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
            # Whether model uses pitch guidance ( f0 ) or not
            pitch = pitch if pitch_guidance else None
            pitchf = pitchf if pitch_guidance else None

        # Prepare data for accelerator ( Move to the right device )
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = accelerator.prepare(
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
            )


            
        # ---------------------    GENERATOR   ---------------------
            model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output

            mel = spec_to_mel_torch(
                spec,
                config.data.filter_length,
                config.data.n_mel_channels,
                config.data.sampling_rate,
                config.data.mel_fmin,
                config.data.mel_fmax,
            )

            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )

            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                config.data.filter_length,
                config.data.n_mel_channels,
                config.data.sampling_rate,
                config.data.hop_length,
                config.data.win_length,
                config.data.mel_fmin,
                config.data.mel_fmax,
            )

            wave = commons.slice_segments(
                wave,
                ids_slice * config.data.hop_length,
                config.train.segment_size,
                dim=3,
            )


            # ---------------------    DISCRIMINATORS   ---------------------
                # Run the discriminator MPD:
            y_dm_hat_r, y_dm_hat_g, _, _ = net_d_mpd(wave, y_hat.detach())
            loss_disc_m, losses_disc_m_r, losses_disc_m_g = discriminator_loss(y_dm_hat_r, y_dm_hat_g)

                # Run the discriminator CQT:
            y_dc_hat_r, y_dc_hat_g, _, _ = net_d_cqt(wave, y_hat.detach())
            loss_disc_c, losses_disc_c_r, losses_disc_c_g = discriminator_loss(y_dc_hat_r, y_dc_hat_g)

                # Run the discriminator STFT:
            y_ds_hat_r, y_ds_hat_g, _, _ = net_d_stft(wave, y_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)


        # Combine Discriminators' loss to obtain Total Discriminators' loss:
            loss_disc_all = loss_disc_m + loss_disc_c + loss_disc_s
            running_loss_disc += loss_disc_all  # Accumulate running losses for Discriminators
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            loss_disc_all_gat = process_gathered_loss("loss_disc_all", loss_disc_all, accelerator)
            running_loss_disc_gat = process_gathered_loss("running_loss_disc", running_loss_disc, accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


        # Zero the grads:
            optim_d_mpd.zero_grad()   # for MPD
            optim_d_cqt.zero_grad()   # for CQT
            optim_d_stft.zero_grad()  # for STFT


        # Backward for MPD:
            accelerator.backward(loss_disc_m)
        # Backward for CQT:
            accelerator.backward(loss_disc_c)
        # Backward for STFT:
            accelerator.backward(loss_disc_s)


        # Clip/norm the gradients for Discriminators:
            grad_norm_d_mpd = commons.clip_grad_value(net_d_mpd.parameters(), None)
            grad_norm_d_cqt = commons.clip_grad_value(net_d_cqt.parameters(), None)
            grad_norm_d_stft = commons.clip_grad_value(net_d_stft.parameters(), None)
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            grad_norm_d_mpd_gat = process_gathered_loss("grad_norm_d_mpd", torch.tensor(grad_norm_d_mpd), accelerator)
            grad_norm_d_cqt_gat = process_gathered_loss("grad_norm_d_cqt", torch.tensor(grad_norm_d_cqt), accelerator)
            grad_norm_d_stft_gat = process_gathered_loss("grad_norm_d_stft", torch.tensor(grad_norm_d_stft), accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


        # Step for Discriminators' optimizers:
            optim_d_mpd.step()
            optim_d_cqt.step()
            optim_d_stft.step()


            # ----------------     Loss Calculation and Backpropagation for Generator     ----------------

            y_dm_hat_r, y_dm_hat_g, fmap_m_r, fmap_m_g = net_d_mpd(wave, y_hat) # MPD Loss
            y_dc_hat_r, y_dc_hat_g, fmap_c_r, fmap_c_g = net_d_cqt(wave, y_hat) # CQT Loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = net_d_stft(wave, y_hat) # STFT Loss

        # Loss functions for gen:


            loss_mel = fn_mel_loss_multiscale(wave, y_hat) * 15 # MEL loss
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            loss_mel_gat = process_gathered_loss("loss_mel", loss_mel, accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


            loss_kl = (kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl) # Kullback–Leibler divergence loss
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            loss_kl_gat = process_gathered_loss("loss_kl", loss_kl, accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


            loss_fm_m = feature_loss(fmap_m_r, fmap_m_g) # Feature matching loss for MPD
            loss_fm_c = feature_loss(fmap_c_r, fmap_c_g) # Feature matching loss for CQT
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) # Feature matching loss for STFT
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            loss_fm_m_gat = process_gathered_loss("loss_fm_m", loss_fm_m, accelerator)
            loss_fm_c_gat = process_gathered_loss("loss_fm_c", loss_fm_c, accelerator)
            loss_fm_s_gat = process_gathered_loss("loss_fm_s", loss_fm_s, accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


            loss_gen_m, losses_gen_m = generator_loss(y_dm_hat_g) # Generator's minimax for MPD
            loss_gen_c, losses_gen_c = generator_loss(y_dc_hat_g) # Generator's minimax for CQT
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g) # Generator's minimax for STFT

        # Combine the Generator's losses for unified "loss_g_total":
            loss_gen_all = loss_gen_m + loss_gen_c + loss_gen_s + loss_fm_m + loss_fm_c + loss_fm_s + loss_mel + loss_kl
            running_loss_gen += loss_gen_all
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            loss_gen_all_gat = process_gathered_loss("loss_gen_all", loss_gen_all, accelerator)
            running_loss_gen_gat = process_gathered_loss("running_loss_gen", running_loss_gen, accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


        # Zero the grad:
            optim_g.zero_grad()

        # Backward for Generator:
            accelerator.backward(loss_gen_all)

        # Clip/norm the gradient for Generator:
            grad_norm_g = commons.clip_grad_value(net_g.parameters(), None)
        #  ---------------    if n_gpus above 1   -------------------------------------------------------------------------------  #
            grad_norm_g_gat = process_gathered_loss("grad_norm_g", torch.tensor(grad_norm_g), accelerator)
        #  ----------------------------------------------------------------------------------------------------------------------  #


        # Step for Generator's optimizer:
            optim_g.step()

        # Update the step and the progress bar
            global_step += 1
            pbar.update(1)


        # Logging of the averaged loss every N mini-batches
            if accelerator.is_main_process and (batch_idx + 1) % N == 0:
                avg_loss_gen = running_loss_gen_gat / N # For Generator
                avg_loss_disc = running_loss_disc_gat / N # For Discriminator
                writer.add_scalar('Loss/Generator_Avg', avg_loss_gen, global_step)
                writer.add_scalar('Loss/Discriminator_Avg', avg_loss_disc, global_step)
            # Resets the running loss counters ( local )
                running_loss_gen = 0.0
                running_loss_disc = 0.0
            # Resets the running loss counters ( global )
                running_loss_gen_gat = 0.0
                running_loss_disc_gat = 0.0

    # Logging and checkpointing
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:

        lr = optim_g.param_groups[0]["lr"]
#        if loss_mel > 75:
#            loss_mel = 75
#        if loss_kl > 9:
#            loss_kl = 9

            # Codename;0's feature: mel spectrogram similarity
        mel_spec_similarity = mel_spectrogram_similarity(y_hat_mel, y_mel)
        accelerator.print(f'Mel Spectrogram Similarity: {mel_spec_similarity:.2f}%')
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_spec_similarity, global_step)


        scalar_dict = {
            "loss/g/total": loss_gen_all_gat,
            "loss/d/total": loss_disc_all_gat,
            "grad_norm/d_mpd": grad_norm_d_mpd_gat,
            "grad_norm/d_cqt": grad_norm_d_cqt_gat,
            "grad_norm/d_stft": grad_norm_d_stft_gat,
            "grad_norm/g": grad_norm_g_gat,
            "loss/g/fm_MPD": loss_fm_m_gat,
            "loss/g/fm_CQT": loss_fm_c_gat,
            "loss/g/fm_STFT": loss_fm_s_gat,
            "loss/g/mel": loss_mel_gat,
            "loss/g/kl": loss_kl_gat,
            "learning_rate": lr,
        }

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        # Generates a reference audio sample during training for evaluation purposes
        with torch.no_grad():
            o, *_ = accelerator.unwrap_model(net_g).infer(*reference)

        audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]}


        summarize(
            writer=writer,
            global_step=global_step,
            images=image_dict,
            scalars=scalar_dict,
            audios=audio_dict,
            audio_sample_rate=sample_rate,
        )

        # Save checkpoint
        model_add = []
        done = False

        if rank == 0:
            # Extract learning rates from optimizers
            lr_g = optim_g.param_groups[0]['lr']
            lr_d_mpd = optim_d_mpd.param_groups[0]['lr']
            lr_d_cqt = optim_d_cqt.param_groups[0]['lr']
            lr_d_stft = optim_d_stft.param_groups[0]['lr']

        # Save weights every N epochs
        if epoch % save_every_epoch == 0 and accelerator.is_main_process:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            save_checkpoint(
                [accelerator.unwrap_model(net_g)],
                [optim_g],
                [lr_g],
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
            )
            save_checkpoint(
                [   # Unwrap each discriminator
                    accelerator.unwrap_model(net_d_mpd), 
                    accelerator.unwrap_model(net_d_cqt), 
                    accelerator.unwrap_model(net_d_stft)
                ],  
                [optim_d_mpd, optim_d_cqt, optim_d_stft],  # Optimizers
                [lr_d_mpd, lr_d_cqt, lr_d_stft],
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
            )
            if custom_save_every_weights and accelerator.is_main_process:
                model_add.append(os.path.join(experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"))

        # Check for completion
        if epoch >= custom_total_epoch and accelerator.is_main_process:
            # Logging
            accelerator.print(f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen.")
            # Saving the weight model
            model_add.append(os.path.join(experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"))

        if model_add and accelerator.is_main_process:
            ckpt = accelerator.get_state_dict(net_g)
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        pitch_guidance=pitch_guidance== True,   # converting 1/0 to True/False,
                        name=model_name,
                        model_dir=m,
                        epoch=epoch,
                        step=global_step,
                        version=version,
                        hps=hps,
                    )

        if accelerator.is_main_process:
            record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
            accelerator.print(record)
        if done:
            accelerator.end_training()
            os._exit(2333333)


if __name__ == "__main__":
    main()