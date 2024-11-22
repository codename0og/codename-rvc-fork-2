import os
import re
import sys
import glob
import json
import torch
import datetime
import math
from typing import Tuple

from distutils.util import strtobool
from random import randint, shuffle
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
    generator_loss,
    kl_loss,
    zero_mean_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from infer.lib.train.extract_model import extract_model

from rvc.layers.synthesizers import Synthesizer

from rvc.layers.algorithm import commons
from rvc.layers.utils import (
    slice_on_last_dim,
    total_grad_norm,
)

from infer.lib.train.mel_processing import MultiScaleMelSpectrogramLoss

# Parse command line arguments
hps = get_hparams()

model_name = hps.name # hparams.model_dir = hparams.experiment_dir = experim
save_every_epoch = hps.save_every_epoch
total_epoch = hps.total_epoch
pretrainG = hps.pretrainG
pretrainD = hps.pretrainD
version = hps.version
gpus = hps.gpus
batch_size = hps.train.batch_size
sample_rate = hps.data.sampling_rate
# Print the sample rate to check its value
# print(f"Sample Rate in get_hparams(): {sample_rate}")
pitch_guidance = hps.if_f0
save_only_latest = hps.if_latest
save_every_weights = hps.save_every_weights
cache_data_in_gpu = hps.if_cache_data_in_gpu

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "0_gt_wavs")

with open(config_save_path, "r") as f:
    config = json.load(f)
config = HParams(**config)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")


# Uncomment to enable TensorFloat32 ( tf32 ) if you wanna experiment ~ Supports only ampere and higher.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Deterministic set to 'True' = Grants reproducibility. You should ideally keep it as 'True' cause I can't promise you'll get decent results without it ~
torch.backends.cudnn.deterministic = True
# Benchmark set to 'True' = Does a benchmark to find the most optimal in performance algorithms/
torch.backends.cudnn.benchmark = False






#from rvc.layers.discriminators.sub.__init__ import (
#    MultiScaleSTFTDiscriminator,
#    MultiScaleSubbandCQTDiscriminator
#)
#from rvc.layers.discriminators.discriminator import CombinedDiscriminator


from rvc.layers.discriminators.sub.mssbcqtd import MultiScaleSubbandCQTDiscriminator

mssbcqt_disc = MultiScaleSubbandCQTDiscriminator(
    sample_rate=sample_rate# hps.data.sampling_rate
)


global_step = 0

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
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

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
    global global_step


    if rank == 0:
        writer = SummaryWriter(log_dir=experiment_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))

    dist.init_process_group(
        backend="gloo",
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
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Initialize models and optimizers
    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=pitch_guidance == True,  # converting 1/0 to True/False
        is_half=config.train.fp16_run and device.type == "cuda",
        sr=sample_rate,
    ).to(device)



#    filters = 64 # 32 stock ( for 24khz )

#    MultiDisc = [
#        MultiScaleSTFTDiscriminator(filters=filters),  # Pass 'filters' to STFT discriminator
#        MultiScaleSubbandCQTDiscriminator()
#    ]

#    net_d = CombinedDiscriminator(MultiDisc)
#    if torch.cuda.is_available():
#        net_d = net_d.to(device)



    net_d = mssbcqt_disc
    if torch.cuda.is_available():
        net_d = net_d.to(device)

    # Define optimizers
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        lr = 1e-5,
        betas = (0.9, 0.999),
        eps = 1e-8,
        #weight_decay = 1e-6,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr = 1e-5,
        betas = (0.9, 0.999),
        eps = 1e-8,
        #weight_decay = 1e-6,
    )
    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    elif torch.cuda.is_available():
        net_g = net_g.to(device)
        net_d = net_d.to(device)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # Use XPU if available (experimental for Intel GPUs)
        net_g = net_g.to('xpu')
        net_d = net_d.to('xpu')
    else:
        # CPU only (no need to move as they're already on CPU)
        net_g = net_g
        net_d = net_d

    # Load checkpoint if available
    try:
        print("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except:
        epoch_str = 1
        global_step = 0
        if pretrainG != "":
            if rank == 0:
                print(f"Loaded pretrained (G) '{pretrainG}'")
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(
                    torch.load(pretrainG, map_location="cpu")["model"]
                )
            else:
                net_g.load_state_dict(
                    torch.load(pretrainG, map_location="cpu")["model"]
                )

        if pretrainD != "":
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(
                    torch.load(pretrainD, map_location="cpu")["model"]
                )
            else:
                net_d.load_state_dict(
                    torch.load(pretrainD, map_location="cpu")["model"]
                )

    # Initialize schedulers and scaler
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
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
                [net_g, net_d],
                [optim_g, optim_d],
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
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_loader, None],
                None,
                cache,
                custom_save_every_weights,
                custom_total_epoch,
                device,
                reference,
            )
        scheduler_g.step()
        scheduler_d.step()


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
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
        sample_rate=sample_rate # hps.data.sampling_rate
    )

    net_g.train()
    net_d.train()

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
            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(
                    spec,
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sampling_rate, #config.data.sample_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                #print("Target Mel Spectrogram Shape:", mel.shape)
                y_mel = commons.slice_segments( # old approach without dim: y_mel = slice_on_last_dim(
                    mel,
                    ids_slice,
                    config.train.segment_size // config.data.hop_length,
                    dim=3,
                )
                # Print shape of y_hat before squeezing
                #print("y_hat Shape Before Squeeze:", y_hat.shape)
                with autocast(enabled=False, dtype=torch.bfloat16):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        config.data.filter_length,
                        config.data.n_mel_channels,
                        config.data.sampling_rate, #config.data.sample_rate,
                        config.data.hop_length,
                        config.data.win_length,
                        config.data.mel_fmin,
                        config.data.mel_fmax,
                    )
                    #print("Generated Mel Spectrogram Shape:", y_hat_mel.shape)
                if use_amp:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments( # old approach without dim: y_mel = slice_on_last_dim(
                    wave,
                    ids_slice * config.data.hop_length,
                    config.train.segment_size,
                    dim=3,
                )
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast(enabled=False, dtype=torch.bfloat16):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
            # Discriminator backward and update
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)

            # grad_norm_d = total_grad_norm(net_d.parameters()) # debugging only

            #grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=0.5) # torch approach
  
            grad_norm_d = commons.clip_grad_value(net_d.parameters(), None) # custom approach

            if math.isnan(grad_norm_d):
                print('grad_norm_d is NaN')
            elif math.isinf(grad_norm_d):
                print('grad_norm_d is Inf')

            #for name, param in net_d.named_parameters():
            #    if param.grad is not None:
            #        grad_value = torch.norm(param.grad.detach(), 2)
            #        if torch.isinf(grad_value):
            #            print(f"Discriminator parameter {name} has an Inf gradient norm: {grad_value}")
            #        elif torch.isnan(grad_value):
            #            print(f"Discriminator parameter {name} has a NaN gradient norm")
            #        else:
            #            print(f"Discriminator parameter {name} gradient norm: {grad_value.item()}")

            scaler.step(optim_d)

            # Generator backward and update
            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False, dtype=torch.bfloat16):

                    #Losses for Generator
                    loss_mel = fn_mel_loss_multiscale(y_hat, wave) * 15 # MEL loss ( wave = GT / reference, y_hat = predicted / generated )
                    loss_kl = (kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl) # KL loss
                    loss_fm = feature_loss(fmap_r, fmap_g) # FM loss
                    loss_zm = zero_mean_loss(y_hat, 0.1) # ZeroMean loss ( dc offset related )

                    loss_gen, losses_gen = generator_loss(y_d_hat_g) # Generator's minimax
                    # loss_gen, losses_gen = generator_loss_sp(y_d_hat_g, mel_generated=y_hat_mel)   # Alternative Generator's loss with added silence penalty

                    # Summed loss of generator's applied losses: gen ( scoring from discriminator ) + FM + MEL + KL
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_zm

            # Backpropagation and optimization
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            
            #grad_norm_g = total_grad_norm(net_g.parameters())      # DEBUGGING ONLY

            grad_norm_g = commons.clip_grad_value(net_g.parameters(), None) # custom clipping approach 

            if math.isnan(grad_norm_g):
                print('grad_norm_g is NaN')
            elif math.isinf(grad_norm_g):
                print('grad_norm_g is Inf')
            
            # DEBUGGING PURPOSES:
            #for name, param in net_g.named_parameters():
            #    if param.grad is not None:
            #        grad_value = torch.norm(param.grad.detach(), 2)
            #        if torch.isinf(grad_value):
            #            print(f"Generator parameter {name} has an Inf gradient norm: {grad_value}")
            #        elif torch.isnan(grad_value):
            #            print(f"Generator parameter {name} has a NaN gradient norm")
            #        else:
            #            print(f"Generator parameter {name} gradient norm: {grad_value.item()}")
                        
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
            "loss/d/total": loss_disc,
            "learning_rate": lr,
            "grad_norm_d": grad_norm_d,
            "grad_norm_g": grad_norm_g,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
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
        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
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