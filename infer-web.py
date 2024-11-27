import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from infer.modules.vc import VC
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from configs import Config
from sklearn.cluster import MiniBatchKMeans
import torch, platform
import re
import numpy as np
import gradio as gr
import faiss
import pathlib
import json
from time import sleep
from subprocess import Popen
import subprocess
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
import signal
import time
from infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)

os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/onnx_models"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "audios"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "dataset"), exist_ok=True)

os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)

if not config.nocheck:
    from infer.lib.rvcmd import check_all_assets, download_all_assets

    if not check_all_assets(update=config.update):
        if config.update:
            download_all_assets(tmpdir=tmp)
            if not check_all_assets(update=config.update):
                logging.error("counld not satisfy all assets needed.")
                exit(1)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    import fairseq

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "Unfortunately, your gpu's not supported for training models."
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")
audio_root = "audios"

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []


def lookup_indices(index_root):
    global index_paths
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


lookup_indices(index_root)
lookup_indices(outside_index_root)

global indexes_list
indexes_list = []

audio_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

for root, dirs, files in os.walk(audio_root, topdown=False):
    for name in files:
        audio_paths.append("%s/%s" % (root, name))



def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''
        

def get_index():
    if check_for_name() != '':
        chosen_model = sorted(names)[0].split(".")[0]
        logs_path="index_root"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file).replace('\\','/')
            return ''
        else:
            return ''

def get_indexes():
    for dirpath, dirnames, filenames in os.walk("index_root"):
        for filename in filenames:
            if filename.endswith(".index") and "trained" not in filename:
                indexes_list.append(os.path.join(dirpath, filename).replace("\\", "/"))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''


def change_choices():
    # Initialize lists to store file names and paths
    names = []
    
    index_paths = []
    audio_paths = []

    # Populate 'names' list with .pth files in 'weight_root'
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)


    # Populate 'index_paths' list with .index files in 'index_root'
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(os.path.join(root, name))

    # Populate 'audio_paths' list with audio files in 'audios' directory
    audios_path = os.path.abspath(os.getcwd()) + "/audios/"
    for file in os.listdir(audios_path):
        audio_paths.append(os.path.join(audio_root, file))

    # Return dictionaries with sorted lists of choices and metadata
    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(index_paths), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p, spk_id5):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f %s' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        config.preprocess_per,
        spk_id5,
    )
    logger.info("Execute: " + cmd)
    
    # Run the command
    p = Popen(cmd, shell=True)
    
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()

    # Polling log output
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


logger = logging.getLogger(__name__)

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe, echl, spk_id5):
    spk_id5 = int(spk_id5)  # Convert to integer
    logger.info(f"Received spk_id5 (total speakers): {spk_id5}")
    print(f"Received spk_id5 (total speakers): {spk_id5}")

    now_dir = os.getcwd()

    # Split GPU IDs (ensure gpus is a list of strings)
    gpus = gpus.split("-")

    # Ensure `gpus_rmvpe` is processed only once
    if gpus_rmvpe != "-":
        gpus_rmvpe_list = gpus_rmvpe.split("-")  # Convert `gpus_rmvpe` to a list
    else:
        gpus_rmvpe_list = []  # Empty list if `gpus_rmvpe` is "-"

    # Sequentially process each speaker directory
    for sid in range(spk_id5):  # Process `sid_0` to `sid_(spk_id5 - 1)`
        logger.info(f"Processing speaker: sid_{sid}")
        print(f"Processing speaker: sid_{sid}")

        # Construct experiment directory for current `sid_n`
        current_exp_dir = os.path.join(exp_dir, f"sid_{sid}")
        current_exp_dir = os.path.normpath(current_exp_dir)  # Normalize path

        # Construct the full log directory path (ensure absolute paths)
        log_dir = os.path.join(now_dir, "logs", current_exp_dir)
        log_dir = os.path.normpath(log_dir)  # Normalize log directory path

        # Create the log directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)

        # Log for debugging
        logger.info(f"Experiment directory set to: {log_dir}")
        print(f"Experiment directory set to: {log_dir}")

        # Create the expanded log file path
        expanded_log_file = os.path.join(log_dir, "extract_f0_feature.log")
        expanded_log_file = os.path.normpath(expanded_log_file)  # Normalize file path
        with open(expanded_log_file, "w") as f:
            pass  # Create empty log file

        def read_log():
            """Generator to read log file continuously."""
            while True:
                with open(expanded_log_file, "r") as f:
                    yield f.read()
                time.sleep(1)
                if done[0]:
                    break

        # Process extraction depending on `f0method`
        if if_f0:
            if f0method != "rmvpe_gpu":
                f0_script_path = os.path.join(now_dir, "infer", "modules", "train", "extract", "extract_f0_print.py")
                f0_script_path = os.path.normpath(f0_script_path)  # Normalize path

                if not os.path.exists(f0_script_path):
                    logger.error(f"F0 script not found: {f0_script_path}")
                    raise FileNotFoundError(f"F0 script not found: {f0_script_path}")

                cmd = (
                    f'"{config.python_cmd}" "{f0_script_path}" '
                    f'"{log_dir}" {n_p} {f0method} {echl}'
                )
                logger.info(f"Execute: {cmd}")
                p = subprocess.Popen(cmd, shell=True, cwd=now_dir)
                done = [False]
                threading.Thread(target=if_done, args=(done, p)).start()
            else:
                if gpus_rmvpe_list:  # If GPUs are specified for RMVPE
                    leng = len(gpus_rmvpe_list)
                    ps = []
                    for idx, n_g in enumerate(gpus_rmvpe_list):
                        f0_rmvpe_script_path = os.path.join(now_dir, "infer", "modules", "train", "extract", "extract_f0_rmvpe.py")
                        f0_rmvpe_script_path = os.path.normpath(f0_rmvpe_script_path)  # Normalize path

                        if not os.path.exists(f0_rmvpe_script_path):
                            logger.error(f"F0 RMVPE script not found: {f0_rmvpe_script_path}")
                            raise FileNotFoundError(f"F0 RMVPE script not found: {f0_rmvpe_script_path}")

                        cmd = (
                            f'"{config.python_cmd}" "{f0_rmvpe_script_path}" '
                            f'{leng} {idx} {n_g} "{log_dir}" {config.is_half}'
                        )
                        logger.info(f"Execute: {cmd}")
                        p = subprocess.Popen(cmd, shell=True, cwd=now_dir)
                        ps.append(p)
                    done = [False]
                    threading.Thread(target=if_done_multi, args=(done, ps)).start()
                else:
                    f0_rmvpe_dml_script_path = os.path.join(now_dir, "infer", "modules", "train", "extract", "extract_f0_rmvpe_dml.py")
                    f0_rmvpe_dml_script_path = os.path.normpath(f0_rmvpe_dml_script_path)  # Normalize path

                    if not os.path.exists(f0_rmvpe_dml_script_path):
                        logger.error(f"F0 RMVPE DML script not found: {f0_rmvpe_dml_script_path}")
                        raise FileNotFoundError(f"F0 RMVPE DML script not found: {f0_rmvpe_dml_script_path}")

                    cmd = (
                        f'{config.python_cmd} "{f0_rmvpe_dml_script_path}" '
                        f'"{log_dir}"'
                    )
                    logger.info(f"Execute: {cmd}")
                    p = subprocess.Popen(cmd, shell=True, cwd=now_dir)
                    p.wait()
                    done = [True]

            # Continuously yield log updates
            yield from read_log()

            with open(expanded_log_file, "r") as f:
                log = f.read()
            logger.info(log)
            yield log

        # Run feature extraction
        leng = len(gpus)
        ps = []
        for idx, n_g in enumerate(gpus):
            feature_script_path = os.path.join(now_dir, "infer", "modules", "train", "extract", "extract_feature_print.py")
            feature_script_path = os.path.normpath(feature_script_path)  # Normalize path

            if not os.path.exists(feature_script_path):
                logger.error(f"Feature extraction script not found: {feature_script_path}")
                raise FileNotFoundError(f"Feature extraction script not found: {feature_script_path}")

            cmd = (
                f'"{config.python_cmd}" "{feature_script_path}" '
                f'{config.device} {leng} {idx} {n_g} "{log_dir}" {version19} {config.is_half}'
            )
            logger.info(f"Execute: {cmd}")
            p = subprocess.Popen(cmd, shell=True, cwd=now_dir)
            ps.append(p)

        done = [False]
        threading.Thread(target=if_done_multi, args=(done, ps)).start()

        # Continuously yield log updates
        yield from read_log()

        with open(expanded_log_file, "r") as f:
            log = f.read()
        logger.info(log)
        yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth doesn't exist. RVC won't use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth doesn't exist. RVC won't use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )



    # Previous 'change_f0' handling mechanism   -   INCOMPATIBLE
#def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
#    path_str = "" if version19 == "v1" else "_v2"
#    return (
#        {"visible": if_f0_3, "__type__": "update"},
#        {"visible": if_f0_3, "__type__": "update"},
#        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
#    )


    # " adapted " old 'if_f0_3' switch mechanism - idk man, tired of this shit, lmao. can't fix it so imma disable the box duh.
def change_f0(
    if_f0_3,
    sr2,
    version19,
    step2b,
    gpus6,
    gpu_info9,
    extraction_crepe_hop_length,
    but2,
    info2,
):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/f0G%s.pth" % (path_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/f0D%s.pth" % (path_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "assets/pretrained%s/f0G%s.pth" % (path_str, sr2),
            "not exist, will not use pretrained model",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "assets/pretrained%s/f0D%s.pth" % (path_str, sr2),
            "not exist, will not use pretrained model",
        )

    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            "assets/pretrained%s/f0G%s.pth" % (path_str, sr2)
            if if_pretrained_generator_exist
            else "",
            "assets/pretrained%s/f0D%s.pth" % (path_str, sr2)
            if if_pretrained_discriminator_exist
            else "",
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )

    return (
        {"visible": False, "__type__": "update"},
        ("assets/pretrained%s/G%s.pth" % (path_str, sr2))
        if if_pretrained_generator_exist
        else "",
        ("assets/pretrained%s/D%s.pth" % (path_str, sr2))
        if if_pretrained_discriminator_exist
        else "",
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])

def get_all_sid_folders(base_dir):
    # Identify all existing 'sid_X' folders
    existing_sids = [
        d for d in os.listdir(base_dir) if d.startswith("sid_") and d[4:].isdigit()
    ]
    if not existing_sids:
        raise FileNotFoundError(f"No 'sid_X' folders found in {base_dir}")

    # Return all sorted existing SID folders
    existing_ids = sorted([int(d[4:]) for d in existing_sids])
    return [f"sid_{id}" for id in existing_ids]


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # Determine the experiment base directory and get all existing SID folders
    exp_dir_base = os.path.join(os.getcwd(), "logs", exp_dir1)
    os.makedirs(exp_dir_base, exist_ok=True)

    # Retrieve all existing SID folders (e.g., sid_0, sid_1, etc.)
    sid_folders = get_all_sid_folders(exp_dir_base)
    logger.info(f"Found SID folders: {sid_folders}")

    # Aggregate entries for filelist from all SID folders
    opt = []
    fea_dim = 256 if version19 == "v1" else 768

    # Iterate through each available SID folder to collect entries
    for sid_folder in sid_folders:
        exp_dir = os.path.join(exp_dir_base, sid_folder)
        spk_id = sid_folder.split("_")[-1]  # Extract speaker ID from folder name

        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs").replace("\\", "/")
        feature_dir = (
            os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")
            .replace("\\", "/")
        )

        # Determine file names for processing
        if if_f0_3:
            f0_dir = os.path.join(exp_dir, "2a_f0").replace("\\", "/")
            f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf").replace("\\", "/")
            names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
            )
        else:
            names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
                [name.split(".")[0] for name in os.listdir(feature_dir)]
            )
        
        # Append entries to the filelist from the current SID folder
        for name in names:
            if if_f0_3:
                opt.append(
                    f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
                    f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
                )
            else:
                opt.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id}")

    # Add mute files (also adjusted)
    mute_dir = os.path.join(os.getcwd(), "logs", "mute").replace("\\", "/")
    for _ in range(2):
        if if_f0_3:
            opt.append(
                f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|"
                f"{mute_dir}/3_feature{fea_dim}/mute.npy|"
                f"{mute_dir}/2a_f0/mute.wav.npy|"
                f"{mute_dir}/2b-f0nsf/mute.wav.npy|{spk_id}"
            )
        else:
            opt.append(f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|{mute_dir}/3_feature{fea_dim}/mute.npy|{spk_id}")
    
    # Shuffle and write the aggregated filelist to filelist.txt
    shuffle(opt)
    with open(os.path.join(exp_dir_base, "filelist.txt"), "w") as f:
        f.write("\n".join(opt))
    
    logger.debug("Write filelist done")
    logger.info(f"Use gpus: {gpus16}")

    # Save configuration file
    config_path = "v1/%s.json" % sr2 if version19 == "v1" or sr2 == "40k" else "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir_base, "config.json")

    if not pathlib.Path(config_save_path).exists():
        config_data = config.json_config.get(config_path, {})
        
        # Update the spk_embed_dim to the number of speakers (length of sid_folders)
        config_data['model']['spk_embed_dim'] = len(sid_folders)

        # Save the updated config to the file
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")
    
    # Construct the training command based on GPU usage
    if gpus16:
        cmd = (
            f'"{config.python_cmd}" infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} '
            f'-f0 {1 if if_f0_3 else 0} -bs {batch_size12} -g {gpus16} '
            f'-te {total_epoch11} -se {save_epoch10} '
            f'{"-pg " + pretrained_G14 if pretrained_G14 else ""} '
            f'{"-pd " + pretrained_D15 if pretrained_D15 else ""} '
            f'-l {1 if if_save_latest13 == "Yes" else 0} '
            f'-c {1 if if_cache_gpu17 == "Yes" else 0} '
            f'-sw {1 if if_save_every_weights18 == "Yes" else 0} '
            f'-v {version19}'
        )
    else:
        cmd = (
            f'"{config.python_cmd}" infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} '
            f'-f0 {1 if if_f0_3 else 0} -bs {batch_size12} '
            f'-te {total_epoch11} -se {save_epoch10} '
            f'{"-pg " + pretrained_G14 if pretrained_G14 else ""} '
            f'{"-pd " + pretrained_D15 if pretrained_D15 else ""} '
            f'-l {1 if if_save_latest13 == "Yes" else 0} '
            f'-c {1 if if_cache_gpu17 == "Yes" else 0} '
            f'-sw {1 if if_save_every_weights18 == "Yes" else 0} '
            f'-v {version19}'
        )
    
    logger.info(f"Execute: {cmd}")
    
    # Execute the training process
    process = Popen(cmd, shell=True, cwd=os.getcwd())
    process.wait()

    return "Training complete. Check the logs in the experiment folder."



# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19, spk_id5):

    
    exp_dir = "logs/%s/%s" % (exp_dir1, f"sid_{spk_id5}")
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    index_save_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    faiss.write_index(index, index_save_path)
    infos.append("Successfully built index into" + " " + index_save_path)
    link_target = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        outside_index_root,
        exp_dir1,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
# DISABLED
#    try:
#        link = os.link if platform.system() == "Windows" else os.symlink
#        link(index_save_path, link_target)
#        infos.append("Link index to outside folder" + " " + link_target)
#    except:
#        infos.append(
#            "Link index to outside folder"
#            + " "
#            + link_target
#            + " "
#            + "Fail"
#        )

    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
    echl
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    # step1:处理数据
    yield get_info_str("step1: Your datasets are being processed..")
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    # step2a:提取音高
    yield get_info_str("step2: The Pitch & feature extraction is in progress..")
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
        )
    ]

    # step3a:训练模型
    yield get_info_str("step3a: The model's being trained..")
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
    )
    yield get_info_str(
        "Training complete. You can check the training logs in the console or the 'train.log' file under the experiment folder."
    )

    # step3b:训练索引
    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str("Finished!")


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

    # previous "config.dml" ( --dml arg ) based approach.
#F0GPUVisible = config.dml == False

#def change_f0_method(f0method8):
#    if f0method8 == "rmvpe_gpu":
#        visible = F0GPUVisible
#    else:
#        visible = False
#    return {"visible": visible, "__type__": "update"}


    # no-dml-trigger
def change_f0_method(f0method8):
    if f0method8 == "rmvpe":
        visible = False
    elif f0method8 == "rmvpe_gpu":
        visible = True
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


#### Ported from Mangio's RVC Fork ####
def match_index(sid0):
    picked = False
    # folder = sid0.split('.')[0]

    # folder = re.split(r'. |_', sid0)[0]
    folder = sid0.split(".")[0].split("_")[0]
    # folder_test = sid0.split('.')[0].split('_')[0].split('-')[0]
    parent_dir = "./logs/" + folder
    # print(parent_dir)
    if os.path.exists(parent_dir):
        # print('path exists')
        for filename in os.listdir(parent_dir.replace("\\", "/")):
            if filename.endswith(".index"):
                for i in range(len(indexes_list)):
                    if indexes_list[i] == (
                        os.path.join(("./logs/" + folder), filename).replace("\\", "/")
                    ):
                        # print('regular index found')
                        break
                    else:
                        if indexes_list[i] == (
                            os.path.join(
                                ("./logs/" + folder.lower()), filename
                            ).replace("\\", "/")
                        ):
                            # print('lowered index found')
                            parent_dir = "./logs/" + folder.lower()
                            break
                        # elif (indexes_list[i]).casefold() == ((os.path.join(("./logs/" + folder), filename).replace('\\','/')).casefold()):
                        #    print('8')
                        #    parent_dir = "./logs/" + folder.casefold()
                        #    break
                        # elif (indexes_list[i]) == ((os.path.join(("./logs/" + folder_test), filename).replace('\\','/'))):
                        #    parent_dir = "./logs/" + folder_test
                        #    print(parent_dir)
                        #    break
                        # elif (indexes_list[i]) == (os.path.join(("./logs/" + folder_test.lower()), filename).replace('\\','/')):
                        #    parent_dir = "./logs/" + folder_test
                        #    print(parent_dir)
                        #    break
                        # else:
                        #    #print('couldnt find index')
                        #    continue

                # print('all done')
                index_path = os.path.join(
                    parent_dir.replace("\\", "/"), filename.replace("\\", "/")
                ).replace("\\", "/")
                # print(index_path)
                return (index_path, index_path)
    else:
        #print('nothing found')
        return ('', '')


#### Ported from Mangio's RVC Fork ####
def whethercrepeornah(radio):
    mango = True if radio == 'mangio-crepe' or radio == 'mangio-crepe-tiny' else False

    return ({"visible": mango, "__type__": "update"})


#Change your Gradio Theme here. 👇 👇 👇 👇 Example: " theme='HaleyCH/HaleyCH_Theme' "
with gr.Blocks(title=" Codename-RVC-Fork 🍇 ") as app:
    gr.HTML("<h1> Codename-RVC-Fork 🍇 </h1>")
    gr.Markdown(
        value=
            "This software is open source under the MIT license. The author does not have any control over the software. Those who use the software and disseminate the sounds exported by the software are fully responsible. <br>If you do not agree with this clause, you cannot use or quote any code and files in the software package. See root directory <b>LICENSE</b> for details."
    )
    with gr.Tabs():

        with gr.TabItem("Inference"):
            with gr.Row():
                sid0 = gr.Dropdown(label="Choose your (.pth) model for inference", choices=sorted(names), value="")
                refresh_button = gr.Button(
                    "Refresh models list, index path and audio files",
                    variant="primary",
                )
                clean_button = gr.Button("Unload the model from GPU / RAM memory:", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Please select the ID of speaker",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean")

            with gr.Group():
                gr.Markdown(
                    value="Male model + female audio? try '-12'. Female model + male audio? Try '12'.  If the pitch is off / voice is distorted, you can adjust it to the appropriate range by yourself. Anywhere from -12 to just 12.  i.e.  6. <br>Sometimes '0' works wonders."
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(
                            label="Transpose ( Number of semitones, integers. )", value=0
                        )
                        input_audio0 = gr.Textbox(
                            label=
                                "If you want to manually specify the path to your audio, do it here:"
                            ,value="PATH/TO/YOUR/AUDIO.wav ( .flac, .mp3 etc. )",
                        )
                        input_audio1 = gr.Dropdown(
                            label="Auto-detect dropdown for audio files in 'audios' folder ( In rvc's root):",
                            choices=sorted(audio_paths),
                            value='',
                            interactive=True,
                        )
                        input_audio1.change(fn=lambda:'',inputs=[],outputs=[input_audio0])
                        f0method0 = gr.Radio(
                            label="Pick the pitch / f0 extraction algorithm. More detailed info on these in 'Code's 101' tab. ( tl;dr? mangio-crepe, rmvpe and fcpe are your best picks.",
                            choices=[
                                "pm",
                                "harvest",
                                "crepe",
                                "mangio-crepe",
                                "mangio-crepe-tiny",
                                "rmvpe",
                                "fcpe",
                            ],
                            value="rmvpe",
                            interactive=True,
                        )
                        crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="crepe_hop_length",
                            value=128,
                            interactive=True,
                            visible=False,
                        )
                        f0method0.change(
                            fn=whethercrepeornah,
                            inputs=[f0method0],
                            outputs=[crepe_hop_length],
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=">=3 uses median filtering on the harvest's extracted f0 results. Value represents filter's radius. Use it to reduce breathiness.",
                            value=3,
                            step=1,
                            interactive=True,
                            visible=True,
                        )
                    with gr.Column():
                        file_index1 = gr.Textbox(
                            label="Path to the feature index file. Leave blank to use the selected result from the dropdown:",
                            value="",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index2, input_audio1],
                        )
                                    # Kept for the legacy's sake or if one wants to tinker with it.
                        # file_big_npy1 = gr.Textbox(
                        #     label="F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:",
                        #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Feature Index search ratio. Controls accent and pronunciation strength. If the model was trained on small dataset, setting it too high can result in artifacts. Recommended: 0.3, 0.5, 0,75 ( Setting it to 0 turns off the index. Model will take all it can from the input audio",
                            value=0.50,
                            interactive=True,
                        )
                    with gr.Column():
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resampling the final output. 0 = no resampling.",
                            value=0,
                            step=1,
                            interactive=False,
                            visible=False,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume envelope scaling (RMS). Think of it as peak-compression+norm. Don't touch unless you know what you're doing.",
                            value=0,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Protect unvoiced consonants and breath sounds to avoid 'em artifacting. 0.5 = Max protection. 0 = No protection. ( High protection may reduce the accuracy of index ) Default is '0.33'",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    f0_file = gr.File(label="F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:")
                    but0 = gr.Button("Inference", variant="primary")
                    with gr.Row():
                        vc_output1 = gr.Textbox(label="Output information")
                        vc_output2 = gr.Audio(label="Export audio (click on the three dots in the lower right corner to download)")
                    but0.click(
                        vc.vc_single,
                        [
                            spk_item,
                            input_audio0,
                            input_audio1,
                            vc_transform0,
                            f0_file,
                            f0method0,
                            file_index1,
                            file_index2,
                            # file_big_npy1,
                            index_rate1,
                            filter_radius0,
                            resample_sr0,
                            rms_mix_rate0,
                            protect0,
                            crepe_hop_length
                        ],
                        [vc_output1, vc_output2],
                    )
            with gr.Group(visible=False):
                with gr.Row(visible=False):
                    with gr.Column(visible=False):
                        vc_transform1 = gr.Number(
                            label="Transpose ( Number of semitones, integers. )", value=0, visible=False
                        )
                        opt_input = gr.Textbox(label="Specify the output folder", value="opt", visible=False)
                        f0method1 = gr.Radio(
                            label="Select pitch f0 extraction algorithm.",
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe",
                            interactive=False,
                            visible=False,
                        )
                        
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=">=3 uses median filtering on the harvest's extracted f0 results. Value represents filter's radius. Use it to reduce breathiness.",
                            value=3,
                            step=1,
                            interactive=False,
                            visible=False,
                        )
                    with gr.Column(visible=False):
                        file_index3 = gr.Textbox(
                            label="Path to the feature index file. Leave blank to use the selected result from the dropdown:",
                            value="",
                            interactive=False,
                            visible=False,
                        )
                        file_index4 = gr.Dropdown(
                            label="Auto-detect index path ( Dropdown )",
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=False,
                            visible=False,
                        )
                        # sid0.select(fn=match_index, inputs=[sid0], outputs=[file_index2, file_index4])
                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Feature Index search ratio. Controls accent and pronunciation strength. If the model was trained on small dataset, setting it too high can result in artifacts. Recommended: 0.3, 0.5, 0,75 ( Setting it to 0 turns off the index. Model will take all it can from the input audio",
                            value=1,
                            interactive=False,
                            visible=False,
                        )
                    with gr.Column(visible=False):
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resampling the final output. 0 = no resampling.",
                            value=0,
                            step=1,
                            interactive=False,
                            visible=False,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume envelope scaling (RMS). Think of it as peak-compression+norm. Don't touch unless you know what you're doing.",
                            value=1,
                            interactive=False,
                            visible=False,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Protect unvoiced consonants and breath sounds to prevent artifacts. 0.5 = Max protection. 0 = No protection. ( High protection may reduce the accuracy of index ) ps. When you 'unload the model', it will become blank ( kinda a bug I can't fix ), in that case move the slider or input the default '0.33'",
                            value=0.33,
                            step=0.01,
                            interactive=False,
                            visible=False,
                        )
                    with gr.Column(visible=False):
                        dir_input = gr.Textbox(
                            label="Enter the path to the audio folder to be processed (just go to the file manager address bar and copy it",
                            value=os.path.abspath(os.getcwd()).replace('\\', '/') + "/audios/",
                            visible=False,
                        )
                        inputs = gr.File(
                            file_count="multiple", label="Audio files can also be imported in batch, with one of two options, prioritizing folders for reading.", visible=False
                        )
                    with gr.Row(visible=False):
                        format1 = gr.Radio(
                            label="Format for the exported file",
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=False,
                            visible=False,
                        )
                        but1 = gr.Button("Output", variant="primary")
                        vc_output3 = gr.Textbox(label="Output")
                    # Set button visibility after it's created
                    but1.visible = False
                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                            crepe_hop_length,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4],
                    api_name="infer_change_voice",
                )
        with gr.TabItem("Training"):
            gr.Markdown(
                value="step1: Step 1: Model / Experiment configuration. Model's training files and generally data is stored in 'logs' folder, i.e. 'logs/MyPogModel123 ( Each model has their own folder in logs. ) "
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label="Model ( experiment ) name:", value="MyPogModel123")
                sr2 = gr.Radio(
                    label="Model sample rate",
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label="Pitch f0 guidance. ( Required for singing-purpose model, not required for speech only. ps. Leave it on for multi-purpose model. ) ",
                    choices=[True, False],
                    value=True,
                    interactive=False,
                    visible=False,
                )
                version19 = gr.Radio(
                    label="Model version",
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    #visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label="Number of CPU threads used used for pitch extraction ( CPU mode ) and dataset processing.",
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():
                gr.Markdown(
                    value="Step2a: 1. Creates your model's experiment folder, 2. Is segmenting and normalizing your dataset, 3. Processed samples end up in 2 folders located in your model's folder: 0_gt_wavs ( og. sample rate ) and '1_16k_wavs' ( downsampled to 16khz ). "
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label="Path to your dataset folders ( Each named 'sid_N', where N is number. ex. 'sid_0' for single-speaker model):", value=os.path.abspath(os.getcwd()) + "\\dataset"
                    )
                    spk_id5 = gr.Textbox(
                        label= " Total amount of speakers; Counts them from '0' (sid_0) onwards. ", #
                        value=0,
                        interactive=True,
                        visible=True,
                    )
                    but1 = gr.Button("Dataset processing", variant="primary")
                    info1 = gr.Textbox(label="Output information", value="")
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7, spk_id5],
                        [info1],
                        api_name="train_preprocess",
                    )
            with gr.Group():
                step2b = gr.Markdown(value="Step2b: Extracts pitch and features using CPU ( Mangio-crepe is CPU/GPU whereas RMVPE-gpu is simply gpu-accelerated")
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label="If you wanna use more than 1 gpu for features extraction, input their IDs separated with '-'. i.e.: '0-1-2' will use gpus: 0, 1 and 2.",
                            value=gpus,
                            interactive=True,
                            visible=True,
                        )
                        gpu_info9 = gr.Textbox(
                            label="GPU_Info", value=gpu_info, visible=True
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label="General info; 'mangio-crepe' is just 'crepe' which allows for the change of 'hop_length' - can extract on cpu but it's very slow ( More about crepe itself in info tab ). rmvpe and rmvpe_gpu are the same, just the gpu variant is gpu accelerated ( rmvpe_gpu is the fastest gpu method ). In short, for super clean audio, try mangio-crepe with hop set to 64 or 32 else rmvpe. ",
                            choices=["mangio-crepe", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe",
                            interactive=True,
                        )
                        # Mangio element
                        extraction_crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="crepe_hop_length",
                            value=64,
                            interactive=True,
                            visible=False,
                        )
                        # Mangio element
                        f0method8.change(fn=whethercrepeornah, inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                        gpus_rmvpe = gr.Textbox(
                            label="if you want to use more than 1 GPU for 'rmvpe_gpu' f0 extraction, Input their ( gpus' ) IDs and separate them with '-'. i.e.: '0-1-2' will use gpu 0, 1 and 2.",
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=False,
                        )
                    but2 = gr.Button("Features extraction", variant="primary")
                    info2 = gr.Textbox(label="Output information", value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            exp_dir1,
                            version19,
                            gpus_rmvpe,
                            extraction_crepe_hop_length,
                            spk_id5
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
            with gr.Group():
                gr.Markdown(value="Step3: Fill in the training settings -> Train feature index -> Train model")
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label="Saving frequency for your model. ( i.e.: If you keep it at '2' you'll have ur model saved every 2nd epoch.)",
                        value=1,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=1,
                        maximum=10000,
                        step=1,
                        label="Total amount of epochs / iterations.",
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label="Batch_size per GPU ( If your gpu's vram allows it, try: 8, 10, 12, 14, 16 or anything in-between. Those don't click for ur case or you have a really small dataset? Perhaps try: 2, 3, 4 or 6. )",
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label="Pick 'Yes' if you want to keep all Generator and Discriminator ckpts and 'No' if you want to only keep the latest. ( Keep it as 'yes' else rip your storage ~ ps. non-dev users don't need it. )",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label="Check 'Yes' if you want to cache the dataset directly in gpu's memory else leave at 'No'. ( WARNING: If you're low on vram / dataset is big, do not use this. You'll get vram OOM errors.)",
                        choices=["Yes", "No"],
                        value="No",
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label="Do you wanna save your model at each save-point iteration? ( Good advice lol: Keep it as 'yes'. )",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label="Path to pretrained/base Generator",
                        value="assets/pretrained_v2/f0G48k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label="Path to pretrained/base Disriminator",
                        value="assets/pretrained_v2/f0D48k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                            change_f0,
                            [if_f0_3, sr2, version19, step2b, gpus6, gpu_info9, extraction_crepe_hop_length, but2, info2],
                            [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15, step2b, gpus6, gpu_info9, extraction_crepe_hop_length, but2, info2],
                    )
                    if_f0_3.change(fn=whethercrepeornah, inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                    gpus16 = gr.Textbox(
                        label="If you wanna use more than 1 gpu, input their IDs separated with '-'. i.e.: '0-1-2' will use gpus: 0, 1 and 2.",
                        value=gpus,
                        interactive=True,
                        visible=False,
                    )
                    but3 = gr.Button("Train model", variant="primary")
                    but4 = gr.Button("Train feature index", variant="primary")
                    info3 = gr.Textbox(label="Output information", value="", max_lines=10)
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            #spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        info3,
                        api_name="train_start",
                    )
                    but4.click(train_index, [exp_dir1, version19, spk_id5], info3)
        with gr.TabItem("Model utils"):
            with gr.Group():
                gr.Markdown(value="Models fusion ( Can be used to fuse best FM, MEL or just mix 2 unrelated models etc. )")
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label="Path to model A", value="", interactive=True, placeholder="Path to ur model A."
                    )
                    ckpt_b = gr.Textbox(
                        label="Path to model B", value="", interactive=True, placeholder="Path to ur model B."
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label="Weight / priority / influence of model A",
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label="Model's sample rate (Proper fusion for 32k isn't supported.)",
                        choices=["40k", "48k"],
                        value="48k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label="Does your model have f0 pitch guidance?",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label="Information to be added.",
                        value="",
                        placeholder="i.e.: A mix of models X1 and X2",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save0 = gr.Textbox(
                        label="Name your fused / hybrid model. ( just name, don't add extension.)",
                        value="",
                        placeholder="i.e.: model_X1_model_X2_50_50",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label="Your model's version",
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button("Fuse", variant="primary")
                    info4 = gr.Textbox(label="Output information", value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                        version_2,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )  # def merge(path1,path2,alpha1,sr,f0,info):
            with gr.Group():
                gr.Markdown(
                    value="Modify model's information. ( Supports only small weight models. Those 50 ish mb .pth models.)"
                )
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label="Model path", value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label="Model's information to be added / modified",
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label="Name for the saved modified model. ( aka. Name your model. )",
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button("Modify", variant="primary")
                    info5 = gr.Textbox(label="Output information", value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(value="View model's information someone added in using above 'Information modifier'.")
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label="Model path", value="", interactive=True
                    )
                    but8 = gr.Button("View", variant="primary")
                    info6 = gr.Textbox(label="Output information", value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value="Can help if you only have your model's generator file ( Maybe you didn't set it to save weights / small models) and you want to extract a small model, so, a weight / ' epoch '. "
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label="Path to Model:",
                        value="PATH\TO\MODEl'S\GENERATOR\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label="Save name", value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label="Sample rate for your model",
                        choices=["32k", "40k", "48k"],
                        value="48k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label="Want 'pitch guidance' for your model? 1 is yes, 0 is no.",
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label="Model version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label="Model information to be placed",
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    but9 = gr.Button("Extract", variant="primary")
                    info7 = gr.Textbox(label="Output information", value="", max_lines=8)
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                    info7,
                    api_name="ckpt_extract",
                )

        with gr.TabItem("ONNX Exporter"):
            with gr.Row():
                ckpt_dir = gr.Textbox(label="Path to your RVC ( Pytorch ) model.", value="", interactive=True)
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label="Path for the exported ONNX model.", value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button("Export as onnx model", variant="primary")
            butOnnx.click(
                export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        tab_faq = "Codename-RVC-Fork Info"
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "Codename-RVC-Fork Info":
                    with open("fork.md", "r", encoding="utf8") as f:
                        info = f.read()
                else:
                    with open("fork.md", "r", encoding="utf8") as f:
                        info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

try:
    import signal

    def cleanup(signum, frame):
        signame = signal.Signals(signum).name
        print(f"Got signal {signame} ({signum})")
        app.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    if config.global_link:
        app.queue(max_size=1022).launch(share=True, max_threads=511)
    else:
        app.queue(max_size=1022).launch(
            max_threads=511,
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
except Exception as e:
    logger.error(str(e))