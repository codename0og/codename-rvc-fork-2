import multiprocessing
import os
import sys

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(*sys.argv[1:])
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
spk_id5 = sys.argv[7]  # Added to capture the spk_id5 value
sr_trgt = sr

import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

f = open("%s/preprocess.log" % exp_dir, "a+")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class PreProcess:
    def __init__(self, sr, sr_trgt, exp_dir, spk_id5, per=3.0):  # spk_id5 is added as an argument
        # Dynamically create directories under exp_dir based on spk_id5
        self.spk_exp_dir = os.path.join(exp_dir, f"sid_{spk_id5}")
        self.gt_wavs_dir = os.path.join(self.spk_exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(self.spk_exp_dir, "1_16k_wavs")
        
        # Create the directories
        os.makedirs(self.spk_exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
        
        # Initialize other parameters
        self.slicer = Slicer(
            sr=sr,
            threshold=-100,
            min_length=3000,
            min_interval=1000,
            hop_size=15,
            max_sil_kept=3000,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.sr_trgt = sr_trgt

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return

        # Resample 0_gt -> target samplerate
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=self.sr_trgt, res_type='soxr_vhq'
        )

        # Normalization step
        tmp_audio = (tmp_audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * tmp_audio

        # Save normalized samples to 0_gt wavs folder as 32 bit float
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr_trgt,
            tmp_audio.astype(np.float32),
        )

        # Resample to 16khz
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000, res_type='soxr_vhq'
        )

        # Save normalized and resampled (to 16khz) samples to 16k wavs folder as 32 bit float
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),  # 16k wavs folder
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr_trgt)
            # Apply high-pass filter
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                start = 0
                while start + int(self.per * self.sr_trgt) <= len(audio):
                    tmp_audio = audio[start : start + int(self.per * self.sr_trgt)]
                    self.norm_write(tmp_audio, idx0, idx1)
                    idx1 += 1
                    start += int(self.per * self.sr_trgt)

                # Handle any remaining audio that doesn't fill the full slice length
                if start < len(audio):
                    tmp_audio = audio[start:]
                    self.norm_write(tmp_audio, idx0, idx1)
                    idx1 += 1

            println("%s\t-> Success" % path)
        except:
            println("%s\t-> %s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, spk_id5, per):
    pp = PreProcess(sr, sr_trgt, exp_dir, spk_id5, per)
    println("Starting preprocessing")
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("Preprocessing finished")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, spk_id5, per)
