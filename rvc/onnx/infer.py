import typing
import os

import librosa
import numpy as np
import onnxruntime

from rvc.f0 import (
    PM,
    Harvest,
    Dio,
    F0Predictor,
)


class Model:
    def __init__(
        self,
        path: typing.Union[str, bytes, os.PathLike],
        device: typing.Literal["cpu", "cuda", "dml"] = "cpu",
    ):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml":
            providers = ["DmlExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(path, providers=providers)


class ContentVec(Model):
    def __init__(
        self,
        vec_path: typing.Union[str, bytes, os.PathLike],
        device: typing.Literal["cpu", "cuda", "dml"] = "cpu",
    ):
        super().__init__(vec_path, device)

    def __call__(self, wav: np.ndarray[typing.Any, np.dtype]):
        return self.forward(wav)

    def forward(self, wav: np.ndarray[typing.Any, np.dtype]):
        if wav.ndim == 2:  # double channels
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        wav = np.expand_dims(np.expand_dims(wav, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: wav}
        logits = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)


predictors: typing.Dict[str, F0Predictor] = {
    "pm": PM,
    "harvest": Harvest,
    "dio": Dio,
}


def get_f0_predictor(
    f0_method: str, hop_length: int, sampling_rate: int
) -> F0Predictor:
    return predictors[f0_method](hop_length=hop_length, sampling_rate=sampling_rate)


class RVC(Model):
    def __init__(
        self,
        model_path: typing.Union[str, bytes, os.PathLike],
        hop_len=512,
        vec_path: typing.Union[str, bytes, os.PathLike] = "vec-768-layer-12.onnx",
        device: typing.Literal["cpu", "cuda", "dml"] = "cpu",
    ):
        super().__init__(model_path, device)
        self.vec_model = ContentVec(vec_path, device)
        self.hop_len = hop_len

    def infer(
        self,
        wav: np.ndarray[typing.Any, np.dtype],
        wav_sr: int,
        model_sr: int = 40000,
        sid: int = 0,
        f0_method="dio",
        f0_up_key=0,
    ) -> np.ndarray[typing.Any, np.dtype[np.int16]]:
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_predictor = get_f0_predictor(
            f0_method,
            self.hop_len,
            model_sr,
        )
        org_length = len(wav)
        if org_length / wav_sr > 50.0:
            raise RuntimeError("wav max length exceeded")

        hubert = self.vec_model(librosa.resample(wav, orig_sr=wav_sr, target_sr=16000))
        hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]

        pitchf = f0_predictor.compute_f0(wav, hubert_length)
        pitchf = pitchf * 2 ** (f0_up_key / 12)
        pitch = pitchf.copy()
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch = np.rint(f0_mel).astype(np.int64)

        pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch = pitch.reshape(1, len(pitch))
        ds = np.array([sid]).astype(np.int64)

        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length = np.array([hubert_length]).astype(np.int64)

        out_wav = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()

        out_wav = np.pad(out_wav, (0, 2 * self.hop_len), "constant")

        return out_wav[0:org_length]

    def forward(
        self,
        hubert: np.ndarray[typing.Any, np.dtype[np.float32]],
        hubert_length: int,
        pitch: np.ndarray[typing.Any, np.dtype[np.int64]],
        pitchf: np.ndarray[typing.Any, np.dtype[np.float32]],
        ds: np.ndarray[typing.Any, np.dtype[np.int64]],
        rnd: np.ndarray[typing.Any, np.dtype[np.float32]],
    ) -> np.ndarray[typing.Any, np.dtype[np.int16]]:
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        return (self.model.run(None, onnx_input)[0] * 32767).astype(np.int16)
