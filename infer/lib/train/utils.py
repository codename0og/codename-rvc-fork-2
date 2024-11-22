import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
import shutil

import codecs
import numpy as np
import torch
from scipy.io.wavfile import read
from collections import OrderedDict
import matplotlib.pyplot as plt

MATPLOTLIB_FLAG = False


def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replace parts of the keys in a dictionary.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = (
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def load_checkpoint(checkpoint_path, models, optimizers=None, load_opt=1):
    """
    Load a checkpoint into a list of models and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        models (list[torch.nn.Module]): List of models to load the checkpoint into.
        optimizers (list[torch.optim.Optimizer], optional): List of optimizers to load the state into. Defaults to None.
        load_opt (int, optional): Whether to load the optimizer state. Defaults to 1.
    """
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    # Load checkpoint data
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    # Optionally handle backward-compatible checkpoint (old version with renamed keys)
    if "model_1" not in checkpoint_dict.get("models", {}):
        print("Detected old checkpoint format, replacing keys.")
        checkpoint_dict = replace_keys_in_dict(
            replace_keys_in_dict(
                checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"
            ),
            ".weight_g", ".parametrizations.weight.original0"
        )

    # Load model states
    for i, model in enumerate(models):
        model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        # Check if the model has a state_dict in the checkpoint
        model_key = f"model_{i + 1}"  # models are stored as "model_1", "model_2", ...
        if model_key in checkpoint_dict["models"]:
            new_state_dict = {k: checkpoint_dict["models"][model_key].get(k, v) for k, v in model_state_dict.items()}

            # Load the state_dict into the model
            if hasattr(model, "module"):
                model.module.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"Warning: No checkpoint found for {model_key}")

    # Load optimizer states if specified
    if optimizers and load_opt == 1:
        for i, optimizer in enumerate(optimizers):
            optim_key = f"optimizer_{i + 1}"
            if optim_key in checkpoint_dict["optimizers"]:
                optimizer.load_state_dict(checkpoint_dict["optimizers"][optim_key]["state_dict"])
            else:
                print(f"Warning: No checkpoint found for {optim_key}")

    # Load learning rates (if needed)
    learning_rates = []
    if "learning_rates" in checkpoint_dict:
        learning_rates = [checkpoint_dict["learning_rates"].get(f"learning_rate_{i+1}", 0) for i in range(len(optimizers))]
    
    # Return relevant checkpoint info
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint_dict['iteration']})")
    return models, optimizers, learning_rates, checkpoint_dict["iteration"]




def save_checkpoint(models, optimizers, learning_rates, iteration, checkpoint_path):
    """
    Save the state of multiple models and an optimizer to a single checkpoint file.

    Args:
        models (list[torch.nn.Module]): List of models to save.
        optimizers (list[torch.optim.Optimizer]): List of optimizers to save.
        learning_rate (float): The current learning rate.
        iteration (int): The current iteration.
        checkpoint_path (str): The path to save the checkpoint to.
    """
    # Save model states
    state_dicts = {
        f"model_{i}": (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
        for i, model in enumerate(models, start=1) # Key is "models"
    }


    # Save optimizer states and their learning rates
    optimizer_dicts = {
        f"optimizer_{i}": {"state_dict": optim.state_dict(), "learning_rate": lr} 
        for i, (optim, lr) in enumerate(zip(optimizers, learning_rates), start=1) # Keys are "optimizers" and "learning_rates"
    }


    # Combine into a single checkpoint
    checkpoint_data = {
        "models": state_dicts,
        "iteration": iteration,
        "optimizers": optimizer_dicts,
        "learning_rates": {f"learning_rate_{i}": lr for i, lr in enumerate(learning_rates, start=1)},
    }
    torch.save(checkpoint_data, checkpoint_path)


    # Create a backwards-compatible checkpoint
    old_version_path = checkpoint_path.replace(".pth", "_old_version.pth")
    checkpoint_data = replace_keys_in_dict(
        replace_keys_in_dict(
            checkpoint_data, ".parametrizations.weight.original1", ".weight_v"
        ),
        ".parametrizations.weight.original0",
        ".weight_g",
    )
    torch.save(checkpoint_data, old_version_path)
    os.replace(old_version_path, checkpoint_path)
    print(f"Saved models to '{checkpoint_path}' (epoch {iteration})")



def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Get the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None


def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert a spectrogram to a NumPy array for visualization.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    sample_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sample_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Load filepaths and associated text from a file.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
    """
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


def get_hparams(init=True):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-se",
        "--save_every_epoch",
        type=int,
        required=True,
        help="checkpoint save frequency (epoch)",
    )
    parser.add_argument(
        "-te", "--total_epoch", type=int, required=True, help="total_epoch"
    )
    parser.add_argument(
        "-pg", "--pretrainG", type=str, default="", help="Pretrained Generator path"
    )
    parser.add_argument(
        "-pd", "--pretrainD", type=str, default="", help="Pretrained Discriminator path"
    )
    parser.add_argument("-g", "--gpus", type=str, default="0", help="split by -")
    parser.add_argument(
        "-bs", "--batch_size", type=int, required=True, help="batch size"
    )
    parser.add_argument(
        "-e", "--experiment_dir", type=str, required=True, help="experiment dir"
    )  # -m
    parser.add_argument(
        "-sr", "--sample_rate", type=str, required=True, help="sample rate, 32k/40k/48k"
    )
    parser.add_argument(
        "-sw",
        "--save_every_weights",
        type=str,
        default="0",
        help="save the extracted model in weights directory when saving checkpoints",
    )
    parser.add_argument(
        "-v", "--version", type=str, required=True, help="model version"
    )
    parser.add_argument(
        "-f0",
        "--if_f0",
        type=int,
        required=True,
        help="use f0 as one of the inputs of the model, 1 or 0",
    )
    parser.add_argument(
        "-l",
        "--if_latest",
        type=int,
        required=True,
        help="if only save the latest G/D pth file, 1 or 0",
    )
    parser.add_argument(
        "-c",
        "--if_cache_data_in_gpu",
        type=int,
        required=True,
        help="if caching the dataset in GPU memory, 1 or 0",
    )

    args = parser.parse_args()
    name = args.experiment_dir
    experiment_dir = os.path.join("./logs", args.experiment_dir)

    config_save_path = os.path.join(experiment_dir, "config.json")
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = "%s/filelist.txt" % experiment_dir
    return hparams

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    """
    A class for storing and accessing hyperparameters.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def copy(self):
        return deepcopy(self)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)