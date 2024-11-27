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
from time import time as ttime
from time import sleep
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


import torch.distributed as dist
import torch.multiprocessing as mp

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from random import randint, shuffle
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

from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)

#from rvc.layers.discriminators import mssbcqtd
from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    #multi_scale_feature_loss,
    #multi_scale_feature_loss_log,
    generator_loss,
    kl_loss,
    #zero_mean_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from infer.lib.train.extract_model import extract_model

from rvc.layers.synthesizers import Synthesizer

from rvc.layers.algorithm import commons
from rvc.layers.utils import (
    slice_on_last_dim,
    total_grad_norm,
)

from infer.lib.train.mel_processing import MultiScaleMelSpectrogramLoss, HighFrequencyArtifactLoss

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


# Uncomment to enable TensorFloat32 ( tf32 ) if you wanna experiment ~ Supports only ampere and higher.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# Deterministic set to 'True' = Grants reproducibility. You should ideally keep it as 'True' cause I can't promise you'll get decent results without it ~
torch.backends.cudnn.deterministic = False
# Benchmark set to 'True' = Does a benchmark to find the most optimal in performance algorithms/
torch.backends.cudnn.benchmark = True


from rvc.layers.discriminators.sub.mpd import MultiPeriodDiscriminatorV2
from rvc.layers.discriminators.sub.mssbcqtd import MultiScaleSubbandCQTDiscriminator
from rvc.layers.discriminators.sub.msstftd import MultiScaleSTFTDiscriminator

#supported_discriminators = {
#    "msstft": MultiScaleSTFTDiscriminator,
#    "mssbcqtd": MultiScaleSubbandCQTDiscriminator,
#}
#discriminators = dict()

#mssbcqt_disc = MultiScaleSubbandCQTDiscriminator(
#    sample_rate=sample_rate# hps.data.sampling_rate
#)


global_step = 0
warmup_epochs = 5
use_warmup = False
warmup_completed = False

from ranger import Ranger

        # Codename;0's tweak / feature
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


import logging
logging.getLogger("torch").setLevel(logging.ERROR)


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
        return f"time={current_time} | training_speed={elapsed_time_str}"


def main():
    """
    Main function to start the training process.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "50000"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        n_gpus = 1
    else:
        device = torch.device("cpu")
        n_gpus = 1
        print("Training with CPU, this will take a long time.")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        # Only spawn processes if multiple GPUs are detected
        # if n_gpus > 1:
        children = []
        for i in range(n_gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    i,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    pitch_guidance,
                    total_epoch,
                    save_every_weights,
                    config,
                    device,
                ),
            )
            children.append(subproc)
            subproc.start()

        for i in range(n_gpus):
            children[i].join()

    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    pitch_guidance,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
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


    if rank == 0:
        writer = SummaryWriter(log_dir=experiment_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1, 
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Create datasets and dataloaders
    #print("Config Data:", config.data)
    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset, # 83666
        num_workers=8, # or 4
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8, # or 4
    )

    # Initialize models and optimizers
    net_g = Synthesizer(
        config.data.filter_length // 2 + 1, # 2048
        config.train.segment_size // config.data.hop_length, # 17280 / 480 = 36 frames
        **config.model,
        use_f0=pitch_guidance == True,  # converting 1/0 to True/False
        is_half=config.train.fp16_run and device.type == "cuda",
        sr=sample_rate,
    ).to(device)



    MPD = MultiPeriodDiscriminatorV2(use_spectral_norm)
    CQT = MultiScaleSubbandCQTDiscriminator(filters=128, sample_rate=sample_rate)
    STFT = MultiScaleSTFTDiscriminator(filters=128)


    # Sets MultiPeriodDiscriminator as primary Discriminator ( net_d_mpd )
    net_d_mpd = MPD
    if torch.cuda.is_available():
        net_d_mpd = net_d_mpd.to(device)

    # Sets MultiScaleSubbandCQTDiscriminator as 1st helper / FM Discriminator ( net_d_cqt )
    net_d_cqt = CQT
    if torch.cuda.is_available():
        net_d_cqt = net_d_cqt.to(device)

    # Sets MultiScaleSTFTDiscriminator as 2nd helper / FM Discriminator ( net_d_stft )
    net_d_stft = STFT
    if torch.cuda.is_available():
        net_d_stft = net_d_stft.to(device)

    # Defining the optimizers
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
    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d_mpd = DDP(net_d_mpd, device_ids=[rank], find_unused_parameters=True)
        net_d_cqt = DDP(net_d_cqt, device_ids=[rank], find_unused_parameters=True)
        net_d_stft = DDP(net_d_stft, device_ids=[rank], find_unused_parameters=True)
    elif torch.cuda.is_available():
        net_g = net_g.to(device)
        net_d_mpd = net_d_mpd.to(device)
        net_d_cqt = net_d_cqt.to(device)
        net_d_stft = net_d_stft.to(device)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # Use XPU if available (experimental for Intel GPUs)
        net_g = net_g.to('xpu')
        net_d_mpd = net_d_mpd.to('xpu')
        net_d_cqt = net_d_cqt.to('xpu')
        net_d_stft = net_d_stft.to('xpu')
    else:
        # CPU only (no need to move as they're already on CPU)
        net_g = net_g
        net_d_mpd = net_d_mpd
        net_d_cqt = net_d_cqt
        net_d_stft = net_d_stft


    # Load checkpoint if available
    try:
        print("Starting training...")
#        _, _, _, epoch_str = load_checkpoint(
#            latest_checkpoint_path(experiment_dir, "D_*.pth"), [net_d_mpd, net_d_cqt, net_d_stft], [optim_d_mpd, optim_d_cqt, optim_d_stft]
#        )

        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), [net_d_mpd, net_d_cqt, net_d_stft], [optim_d_mpd, optim_d_cqt, optim_d_stft]
        )


        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), [net_g], [optim_g]
        )

        #epoch_str = max(epoch_str_g, epoch_str_d)  # Use the maximum epoch from both models
        
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)


    except:
        epoch_str = 1
        global_step = 0

        # Loading the pretrained Generator model
        if pretrainG != "":
            if rank == 0:
                print(f"Loading pretrained (G) '{pretrainG}'")
            # Load the generator model
            checkpoint = torch.load(pretrainG, map_location="cpu")
            state_dicts = checkpoint["models"] # uses key "models"
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(state_dicts["model_1"]) # As "model_1" as there's just 1 generator
            else:
                net_g.load_state_dict(state_dicts["model_1"])

        # Loading the pretrained Discriminator model
        if pretrainD != "":
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            # Load the discriminator models
            checkpoint = torch.load(pretrainD, map_location="cpu")
            state_dicts = checkpoint["models"] # uses key "models"

            # First goes the:  MultiPeriodDiscriminator
            if hasattr(net_d_mpd, "module"):
                net_d_mpd.module.load_state_dict(state_dicts["model_1"]) # "model_1"
            else:
                net_d_mpd.load_state_dict(state_dicts["model_1"])

            # Second goes the:  MultiScale SubBand CQT Discriminator
            if hasattr(net_d_cqt, "module"):
                net_d_cqt.module.load_state_dict(state_dicts["model_2"]) # "model_2"
            else:
                net_d_cqt.load_state_dict(state_dicts["model_2"])

            # Third goes the: MultiScale STFT Discriminator
            if hasattr(net_d_stft, "module"):
                net_d_stft.module.load_state_dict(state_dicts["model_3"]) # "model_3"
            else:
                net_d_stft.load_state_dict(state_dicts["model_3"])


    # Initialize the warmup scheduler only if `use_warmup` is True
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

    scaler = GradScaler(enabled=config.train.fp16_run and device.type == "cuda")

    cache = []
   # get the first sample as reference for tensorboard evaluation
    for info in train_loader:
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        reference = (
            phone.to(device),
            phone_lengths.to(device),
            pitch.to(device) if pitch_guidance else None,
            pitchf.to(device) if pitch_guidance else None,
            sid.to(device),
        )
        break

    for epoch in range(epoch_str, total_epoch + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d_mpd, net_d_cqt, net_d_stft],
                [optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft],
                scaler,
                [train_loader, None],
                [writer, writer_eval],
                cache,
                custom_save_every_weights,
                custom_total_epoch,
                device,
                reference,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d_mpd, net_d_cqt, net_d_stft],
                [optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft],
                scaler,
                [train_loader, None],
                None,
                cache,
                custom_save_every_weights,
                custom_total_epoch,
                device,
                reference,
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
#            decay_scheduler_d_stft.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    scaler,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    reference,
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
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step

    if epoch == 1:
        pass
    net_g, net_d_mpd, net_d_cqt, net_d_stft = nets #     net_g, net_d_mpd, net_d_cqt, net_d_stft = nets
    optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft = optims #     optim_g, optim_d_mpd, optim_d_cqt, optim_d_stft = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)


        # Additional losses:
    #hf_loss = HighFrequencyArtifactLoss(config) # Loss function targetting High-frequency related artifacts ( Which mostly affect sibilants like "s", "c" ) - EXPERIMENTAL

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
        sample_rate=sample_rate # 48khz
    )


    net_g.train()
    net_d_mpd.train()
    net_d_cqt.train()
    net_d_stft.train()

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
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
                cache.append(
                    (
                        batch_idx,
                        (
                            phone.cuda(rank, non_blocking=True),
                            phone_lengths.cuda(rank, non_blocking=True),
                            (
                                pitch.cuda(rank, non_blocking=True)
                                if pitch_guidance
                                else None
                            ),
                            (
                                pitchf.cuda(rank, non_blocking=True)
                                if pitch_guidance
                                else None
                            ),
                            spec.cuda(rank, non_blocking=True),
                            spec_lengths.cuda(rank, non_blocking=True),
                            wave.cuda(rank, non_blocking=True),
                            wave_lengths.cuda(rank, non_blocking=True),
                            sid.cuda(rank, non_blocking=True),
                        ),
                    )
                )
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
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
            if device.type == "cuda" and not cache_data_in_gpu:
                phone = phone.cuda(rank, non_blocking=True)
                phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                pitch = pitch.cuda(rank, non_blocking=True) if pitch_guidance else None
                pitchf = (
                    pitchf.cuda(rank, non_blocking=True) if pitch_guidance else None
                )
                sid = sid.cuda(rank, non_blocking=True)
                spec = spec.cuda(rank, non_blocking=True)
                spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                wave = wave.cuda(rank, non_blocking=True)
                wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
            else:
                phone = phone.to(device)
                phone_lengths = phone_lengths.to(device)
                pitch = pitch.to(device) if pitch_guidance else None
                pitchf = pitchf.to(device) if pitch_guidance else None
                sid = sid.to(device)
                spec = spec.to(device)
                spec_lengths = spec_lengths.to(device)
                wave = wave.to(device)
                wave_lengths = wave_lengths.to(device)

            # Forward pass
            use_amp = config.train.fp16_run and device.type == "cuda"

            with autocast(enabled=use_amp, dtype=torch.bfloat16): # with autocast(enabled=use_amp, dtype=torch.float16 or torch.bfloat16):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)  # Generator's prediction start
                
                mel = spec_to_mel_torch( # Turns spectrogram files of groundtruth audio ( y ) into mel spectrograms
                    spec, # 298 frames 
                    config.data.filter_length, # 2048
                    config.data.n_mel_channels, # 128
                    config.data.sampling_rate, # 48000
                    config.data.mel_fmin, # 0.0
                    config.data.mel_fmax, # null
                )

                y_mel = commons.slice_segments( # Slices / segments the mel spectrograms of groundtruth ( y )
                    mel,
                    ids_slice,
                    config.train.segment_size // config.data.hop_length,
                    dim=3,
                )
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch( # Turns generator's prediction ( y_hat ) into a mel spectrogram representation
                        y_hat.float().squeeze(1),
                        config.data.filter_length,  # 2048
                        config.data.n_mel_channels, # 128
                        config.data.sampling_rate, # 48000
                        config.data.hop_length, # 480
                        config.data.win_length, # 2048
                        config.data.mel_fmin, # 0.0
                        config.data.mel_fmax,   # null
                    )

                if use_amp:
                    y_hat_mel = y_hat_mel.half()

                wave = commons.slice_segments( # Produces ground_truth audio ( y ) waveforms which are segmented ( according to 'ids_slice' and hop_length ) result length: 0.36sec
                    wave, # 2.98 sec
                    ids_slice * config.data.hop_length,
                    config.train.segment_size,
                    dim=3,
                )

                # ----------   Discriminators Update   ----------

                # Zeroing gradients
                optim_d_mpd.zero_grad()   # for MPD
                optim_d_cqt.zero_grad()   # for CQT
                optim_d_stft.zero_grad()  # for STFT


                # Run the discriminator: MPD
                y_dm_hat_r, y_dm_hat_g, _, _ = net_d_mpd(wave, y_hat.detach()) # compares segmented 'wave' to generator's prediction 'y_hat'
                with autocast(enabled=False):
                    loss_disc_m, losses_disc_m_r, losses_disc_m_g = discriminator_loss(
                        y_dm_hat_r, y_dm_hat_g
                    )

                # Run the discriminator: CQT
                y_dc_hat_r, y_dc_hat_g, _, _ = net_d_cqt(wave, y_hat.detach()) # compares segmented 'wave' to generator's prediction 'y_hat'
                with autocast(enabled=False):
                    loss_disc_c, losses_disc_c_r, losses_disc_c_g = discriminator_loss(
                        y_dc_hat_r, y_dc_hat_g
                    )

                # Run the discriminator: STFT
                y_ds_hat_r, y_ds_hat_g, _, _ = net_d_stft(wave, y_hat.detach()) # compares segmented 'wave' to generator's prediction 'y_hat'
                with autocast(enabled=False):
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                        y_ds_hat_r, y_ds_hat_g
                    )

                # Combining the discriminators' loss for unified "loss_d_total"
                loss_disc_all = loss_disc_m + loss_disc_c + loss_disc_s  # triple: loss_disc_m + loss_disc_c + loss_disc_s


            # Backward and Step for: MPD
            scaler.scale(loss_disc_m).backward()
            scaler.unscale_(optim_d_mpd)

            # Backward and Step for: CQT
            scaler.scale(loss_disc_c).backward()
            scaler.unscale_(optim_d_cqt)

            # Backward and Step for: STFT
            scaler.scale(loss_disc_s).backward()
            scaler.unscale_(optim_d_stft)

            # Clip/norm the gradients for Discriminators # 83666
            grad_norm_d_mpd = torch.nn.utils.clip_grad_norm_(net_d_mpd.parameters(), max_norm=1000.0) # alternatively "5.0"
            grad_norm_d_cqt = torch.nn.utils.clip_grad_norm_(net_d_cqt.parameters(), max_norm=1000.0) # alternatively "5.0"
            grad_norm_d_stft = torch.nn.utils.clip_grad_norm_(net_d_stft.parameters(), max_norm=1000.0) # alternatively "5.0"

        # Nan and Inf debugging:

            # for MPD
            if not torch.isfinite(grad_norm_d_mpd):
                print('grad_norm_d_mpd is NaN or Inf')

            # for CQT
            if not torch.isfinite(grad_norm_d_cqt):
                print('grad_norm_d_cqt is NaN or Inf')

            # for STFT
            if not torch.isfinite(grad_norm_d_stft):
                print('grad_norm_d_stft is NaN or Inf')

            scaler.step(optim_d_mpd)
            scaler.step(optim_d_cqt)
            scaler.step(optim_d_stft)

            scaler.update() # Adjust the loss scale based on the applied gradients


            # ----------   Generator Update   ----------

            # Generator backward
            with autocast(enabled=use_amp, dtype=torch.bfloat16): # (enabled=use_amp, dtype=torch.bfloat16)  - If you want to override amp's default 'fp16' with 'bf16'

                # Zeroing gradients
                optim_g.zero_grad() # For Generator

                # MPD Loss:
                _, y_dm_hat_g, fmap_m_r, fmap_m_g = net_d_mpd(wave, y_hat) # y_dm_hat_r

                # CQT Loss:
                _, y_dc_hat_g, fmap_c_r, fmap_c_g = net_d_cqt(wave, y_hat) # y_dm_hat_r

                # STFT Loss:
                _, y_ds_hat_g, fmap_s_r, fmap_s_g = net_d_stft(wave, y_hat) # y_dm_hat_r


            # Loss functions for Generator:
                with autocast(enabled=False): # In full precision  ( FP32 )
                    # MPD:
                    loss_fm_m = feature_loss(fmap_m_r, fmap_m_g) # Feature matching loss for MPD

                    # CQT:
                    loss_fm_c = feature_loss(fmap_c_r, fmap_c_g) # Feature matching loss for CQT

                    # STFT:
                    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) # Feature matching loss for STFT

                    #loss_fm = multi_scale_feature_loss_log(fmap_r, fmap_g) # Multi-scale FM loss with logarithmic approach - EXPERIMENTAL

                    loss_mel = fn_mel_loss_multiscale(wave, y_hat) * 15 # MEL loss
                    loss_kl = (kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl) # Kullback–Leibler divergence loss                    

                    loss_gen_m, _ = generator_loss(y_dm_hat_g) # Generator's minimax for MPD
                    loss_gen_c, _ = generator_loss(y_dc_hat_g) # Generator's minimax for CQT
                    loss_gen_s, _ = generator_loss(y_ds_hat_g) # Generator's minimax for STFT

                    # Summed loss of generator:
                    loss_gen_all = loss_gen_m + loss_fm_m + loss_gen_c + loss_fm_c + loss_gen_s + loss_fm_s + loss_mel + loss_kl

            # Backpropagation and generator optimization
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            
            # Clip/norm the gradients for Generator # 83666
            grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1000.0) # alternatively "5.0"

        # Nan and Inf debugging for Generator
            if not torch.isfinite(grad_norm_g):
                print('grad_norm_g is NaN or Inf')

            scaler.step(optim_g)
            scaler.update()

            global_step += 1
            pbar.update(1)

    # Logging and checkpointing
    if rank == 0:
        lr = optim_g.param_groups[0]["lr"]
#        if loss_mel > 75:
#            loss_mel = 75
#        if loss_kl > 9:
#            loss_kl = 9

            # Codename;0's tweak / feature
        # Calculate the mel spectrogram similarity
        mel_spec_similarity = mel_spectrogram_similarity(y_hat_mel, y_mel)
                
        # Print the similarity percentage to monitor during training
        print(f'Mel Spectrogram Similarity: {mel_spec_similarity:.2f}%')

        # Logging the similarity percentage to TensorBoard
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_spec_similarity, global_step)


        scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc_all,
            "learning_rate": lr,
            "grad/norm_g": grad_norm_g,
            "grad/norm_d_mpd": grad_norm_d_mpd,
            "grad/norm_d_cqt": grad_norm_d_cqt,
            "grad/norm_d_stft": grad_norm_d_stft,
            "loss/g/fm_MPD": loss_fm_m,
            "loss/g/fm_CQT": loss_fm_c,
            "loss/g/fm_STFT": loss_fm_s,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            #"loss/g/hf": loss_hf,
            #"loss/g/zm": loss_zm,
        }
        # commented out
        # scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
        # scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
        # scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        with torch.no_grad():
            if hasattr(net_g, "module"):
                o, *_ = net_g.module.infer(*reference)
            else:
                o, *_ = net_g.infer(*reference)
        audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]}

        summarize(
            writer=writer,
            global_step=global_step,
            images=image_dict,
            scalars=scalar_dict,
            audios=audio_dict,
            audio_sample_rate=sample_rate, #config.data.sampling_rate,
        )

    # Save checkpoint
    model_add = []
    model_del = []
    done = False

    if rank == 0:

        # Extract learning rates from optimizers
        lr_g = optim_g.param_groups[0]['lr']
        lr_d_mpd = optim_d_mpd.param_groups[0]['lr']
        lr_d_cqt = optim_d_cqt.param_groups[0]['lr']
        lr_d_stft = optim_d_stft.param_groups[0]['lr']

        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"

            # Save Generator checkpoint
            save_checkpoint(
                [net_g],
                [optim_g],
                [lr_g],
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
            )
            save_checkpoint(
                [net_d_mpd, net_d_cqt, net_d_stft], #                 [net_d_mpd, net_d_cqt, net_d_stft],
                [optim_d_mpd, optim_d_cqt, optim_d_stft], #                 [optim_d_mpd, optim_d_cqt, optim_d_stft],
                [lr_d_mpd, lr_d_cqt, lr_d_stft], #                 [lr_d_mpd, lr_d_cqt, lr_d_stft],
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
            )

            if custom_save_every_weights:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )


        # Check completion
        if epoch >= custom_total_epoch:
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen."
            )

            # Final model
            model_add.append(
                os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
            )
            done = True

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        pitch_guidance=pitch_guidance
                        == True,  # converting 1/0 to True/False,
                        name=model_name,
                        model_dir=m,
                        epoch=epoch,
                        step=global_step,
                        version=version,
                        hps=hps,
                    )
        # Clean-up old best epochs
        for m in model_del:
            os.remove(m)
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)
        if done:
            os._exit(2333333)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()