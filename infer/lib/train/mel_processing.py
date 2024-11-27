import torch
import torch.utils.data
import torch.nn as nn

from librosa.filters import mel as librosa_mel_fn

import math

import typing
from typing import List

from scipy import signal

import functools

from collections import namedtuple


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    Dynamic range compression using log10.

    Args:
        x (torch.Tensor): Input tensor.
        C (float, optional): Scaling factor. Defaults to 1.
        clip_val (float, optional): Minimum value for clamping. Defaults to 1e-5.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    Dynamic range decompression using exp.

    Args:
        x (torch.Tensor): Input tensor.
        C (float, optional): Scaling factor. Defaults to 1.
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    """
    Spectral normalization using dynamic range compression.

    Args:
        magnitudes (torch.Tensor): Magnitude spectrogram.
    """
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    """
    Spectral de-normalization using dynamic range decompression.

    Args:
        magnitudes (torch.Tensor): Normalized spectrogram.
    """
    return dynamic_range_decompression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """
    Compute the spectrogram of a signal using STFT.

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT window size.
        hop_size (int): Hop size between frames.
        win_size (int): Window size.
        center (bool, optional): Whether to center the window. Defaults to False.
    """
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Convert a spectrogram to a mel-spectrogram.

    Args:
        spec (torch.Tensor): Magnitude spectrogram.
        n_fft (int): FFT window size.
        num_mels (int): Number of mel frequency bins.
        sample_rate (int): Sampling rate of the audio signal.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute the mel-spectrogram of a signal.

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT window size.
        num_mels (int): Number of mel frequency bins.
        sample_rate (int): Sampling rate of the audio signal.
        hop_size (int): Hop size between frames.
        win_size (int): Window size.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        center (bool, optional): Whether to center the window. Defaults to False.
    """
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)

    return melspec


class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default torch.nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = torch.nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = True,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        
        if self.sample_rate == 48000:
            seg_value = 17280
        elif self.sample_rate in [40000, 32000]:
            seg_value = 12800
        else:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        
        self.seg_value = seg_value

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        # mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from:
        # https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        B, C, T = wav.shape

#        print(f"Shape of wav initially: {wav.shape}") #

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            # Default padding values for non-match_stride cases
            right_pad = 0
            pad = 0

        # Trim the excess frames from both ends
        excess_frames = wav.shape[-1] % self.seg_value
#        print("Excess frames from padding:", {excess_frames})

        if excess_frames > 0:
            # Trim excess frames from both sides (half at the front, half at the end)
            front_trim = excess_frames // 2
            end_trim = excess_frames - front_trim  # To handle odd excess frames
            wav = wav[..., front_trim:-end_trim]  # Trim from both sides
#        print("Wave shape after trimming excess frames:", {wav.shape})
        else:
            right_pad = 0
            pad = 0

#        print("Shape of wav before nn.functional.pad:", {wav.shape})
        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")
#        print("Shape of wav after nn.functional.pad:", {wav.shape})

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            # Drop first two and last two frames, which are added
            # because of padding. Now num_frames * hop_length = num_samples.
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sample_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(
                x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(
                y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))

            loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)

        return loss


class HighFrequencyArtifactLoss(nn.Module):
    def __init__(self, config, loss_fn=torch.nn.L1Loss()):
        """
        High-frequency artifact loss that operates directly on mel spectrograms.
        
        Args:
            config (object): Configuration object containing all necessary parameters.
            loss_fn (torch.nn.Module): Loss function to compute the difference between high-frequency components (default: L1Loss).
        """
        super().__init__()
        self.config = config  # Configuration passed during initialization
        
        # Set cutoffs for frequency range in Hz
        self.low_freq_cutoff = 6000  # Default to 6kHz
        self.high_freq_cutoff = 15000  # Default to 15kHz
        self.loss_fn = loss_fn  # Default to L1 loss

    def forward(self, y_hat_mel, y_mel):
        """
        Calculate high-frequency artifact loss between predicted and ground truth mel spectrograms.
        
        Args:
            y_hat_mel (Tensor): Predicted mel spectrogram from the generator.
            y_mel (Tensor): Ground truth mel spectrogram.
        
        Returns:
            Tensor: High-frequency artifact loss value.
        """
        # Apply high-frequency mask to both mel spectrograms
        high_freq_y = self._get_high_freq_mask(y_mel)
        high_freq_y_hat = self._get_high_freq_mask(y_hat_mel)

        # Compute the L1 loss between the high-frequency components
        loss = self.loss_fn(high_freq_y_hat, high_freq_y)
        return loss * 1

    def _get_high_freq_mask(self, mel_spec):
        """
        Apply a mask to retain only the high-frequency components of the mel spectrogram.
        
        Args:
            mel_spec (Tensor): The mel spectrogram to mask.
        
        Returns:
            Tensor: High-frequency components of the mel spectrogram.
        """
        # Get the number of mel bins
        mel_bin = mel_spec.shape[1]  # This is the number of mel bins (n_mels)
        
        # Calculate the frequency bins corresponding to the high-frequency range
        low_bin = int(self.low_freq_cutoff / 24000 * mel_bin)  # Assume fs=48kHz and use normalized bin
        high_bin = int(self.high_freq_cutoff / 24000 * mel_bin)

        # Mask to select high-frequency bins
        high_freq_spec = mel_spec[:, high_bin:, :]  # Select bins above the cutoff
        return high_freq_spec