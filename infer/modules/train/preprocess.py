import multiprocessing
import os
import sys
import traceback

from scipy import signal
from scipy.io import wavfile
import librosa
import numpy as np



now_dir = os.getcwd()
sys.path.append(now_dir)

# Parse command-line arguments
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
spk_id5 = int(sys.argv[7])  # Updated to use as an integer
sr_trgt = sr

# Log file
f = open(f"{exp_dir}/preprocess.log", "a+")

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer


def println(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class PreProcess:
    def __init__(self, sr, sr_trgt, exp_dir, spk_id5, per=3.0):
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
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.sr_trgt = sr_trgt

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            println(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return

        # Resample 0_gt -> target samplerate
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=self.sr_trgt, res_type='soxr_vhq'
        )

        # Normalization step
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio

        # Save normalized samples to 0_gt wavs folder as 32-bit float
        wavfile.write(
            f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav",
            self.sr_trgt,
            tmp_audio.astype(np.float32),
        )

        # Resample to 16kHz
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000, res_type='soxr_vhq'
        )

        # Save normalized and resampled (to 16kHz) samples to 16k wavs folder as 32-bit float
        wavfile.write(
            f"{self.wavs16k_dir}/{idx0}_{idx1}.wav",
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
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
            println(f"{path}\t-> Success")
        except Exception as e:
            println(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                (os.path.join(inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
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
                for p in ps:
                    p.join()
        except Exception as e:
            println(f"Fail. {traceback.format_exc()}")


def preprocess_trainset(inp_root, sr, n_p, exp_dir, spk_id5, per):
    # List all 'sid_N' directories in inp_root
    sid_dirs = sorted(
        [d for d in os.listdir(inp_root) if os.path.isdir(os.path.join(inp_root, d)) and d.startswith("sid_")],
        key=lambda x: int(x.split('_')[1])  # Sort based on the numeric part after 'sid_'
    )

    
    # Validate folder range against spk_id5
    expected_sid_dirs = [f"sid_{i}" for i in range(int(spk_id5))]
    if set(sid_dirs) != set(expected_sid_dirs):
        raise ValueError(f"Expected folders {expected_sid_dirs} but found {sid_dirs}")

    # Process each 'sid_N' folder
    for sid in sid_dirs:
        sid_input_path = os.path.join(inp_root, sid)
        sid_numeric = int(sid.split("_")[1])  # Extract numeric portion of 'sid_N'

        # Instantiate PreProcess class for this specific sid folder
        println(f"Processing {sid}...")
        pp = PreProcess(sr, sr_trgt, exp_dir, sid_numeric, per)  # Pass base exp_dir, sid_numeric for dynamic handling
        
        # Run preprocessing on the current sid folder
        pp.pipeline_mp_inp_dir(sid_input_path, n_p)
        println(f"Finished processing {sid}")

    # Final message after all speakers are processed
    println("All speaker folders processed. You're safe to move onto feature / f0 extraction!")
    
if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, spk_id5, per)