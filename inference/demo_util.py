import torch
import clip
import subprocess
from pathlib import Path
import os
import cv2
import glob
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import librosa
from omegaconf import OmegaConf
import importlib


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int, start_second, truncate_second) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    if truncate_second is None:
        new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}.mp4')
        cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
        cmd += f'-y -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
        subprocess.call(cmd.split())
    else:
        new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}_truncate_{start_second}_{truncate_second}.mp4')
        cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
        cmd += f'-y -ss {start_second} -t {truncate_second} -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
        subprocess.call(cmd.split())
    return new_path


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



class Extract_CAVP_Features(torch.nn.Module):

    def __init__(self, fps=4, batch_size=2, device=None, tmp_path="./", video_shape=(224,224), config_path=None, ckpt_path=None):
        super(Extract_CAVP_Features, self).__init__()
        self.fps = fps
        self.batch_size = batch_size
        self.device = device
        self.tmp_path = tmp_path

        # Initalize Stage1 CAVP model:
        print("Initalize Stage1 CAVP Model")
        config = OmegaConf.load(config_path)
        self.stage1_model = instantiate_from_config(config.model).to(device)

        # Loading Model from:
        assert ckpt_path is not None
        print("Loading Stage1 CAVP Model from: {}".format(ckpt_path))
        self.init_first_from_ckpt(ckpt_path)
        self.stage1_model.eval()
        
        # Transform:
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])
    
    
    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.stage1_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    
    @torch.no_grad()
    def forward(self, frame_dir):
        """
        frame_dir: 프레임 이미지 (frame_000.jpg ~ frame_031.jpg)가 들어있는 디렉토리 경로
        """
        feat_batch_list = []
        video_feats = []

        for idx in range(32):
            frame_filename = f"frame_{idx:03d}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            if not os.path.exists(frame_path):
                print(f"[WARNING] Frame not found: {frame_path}")
                continue

            img = Image.open(frame_path).convert("RGB")
            img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)
            feat_batch_list.append(img_tensor)
                # Forward:
            if len(feat_batch_list) == self.batch_size:
                # Stage1 Model:
                input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                feat_batch_list = []
        else:
            if len(feat_batch_list) != 0:
                input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                video_feats.extend(contrastive_video_feats.detach().cpu().numpy())

            video_contrastive_feats = np.concatenate(video_feats, axis=0)
            return video_contrastive_feats



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)
    model.cuda()
    model.eval()
    return model


def inverse_op(spec):
    sr = 22050
    n_fft = 1024
    fmin = 125
    fmax = 7600
    nmels = 80
    hoplen = 1024 // 4
    spec_power = 1

    # Inverse Transform
    spec = spec * 100 - 100
    spec = (spec + 20) / 20
    spec = 10 ** spec
    spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
    wav = librosa.griffinlim(spec_out, hop_length=hoplen)
    return wav


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def frames_to_video(frame_dir, fps=4):
    """
    frame_dir: 프레임 이미지들이 들어있는 디렉토리 경로
    fps: 생성할 영상의 FPS (기본값: 4)
    
    return: 저장된 영상 파일 경로
    """
    # 프레임 파일 경로 정렬
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if len(frame_paths) == 0:
        raise ValueError(f"No frame images found in {frame_dir}")

    # 프레임 크기 가져오기
    sample_frame = cv2.imread(frame_paths[0])
    height, width, _ = sample_frame.shape

    # 저장 경로 설정
    save_dir = "new_original_video"
    os.makedirs(save_dir, exist_ok=True)

    # 디렉토리 이름을 기반으로 파일 이름 생성
    folder_name = os.path.basename(os.path.normpath(frame_dir))
    video_filename = f"{folder_name}.mp4"
    video_path = os.path.join(save_dir, video_filename)

    # 영상 writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Failed to read frame: {path}")
            continue
        video_writer.write(img)

    video_writer.release()
    print(f"[INFO] Saved video to: {video_path}")
    return video_path