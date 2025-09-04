import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

from diff_foley.util import instantiate_from_config

fps = 4  # CAVP 모델은 fps=4로 세팅되어 있음

def load_stage1_model(cavp_config_path, cavp_ckpt_path, device):
    print(f"Initializing Stage1 CAVP model on {device}...")
    config = OmegaConf.load(cavp_config_path)
    model = instantiate_from_config(config.model).to(device)
    checkpoint = torch.load(cavp_ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Restored model with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_one_video(vid, video_frame_root, audio_mel_root, save_feature_root, model, device):
    frame_dir = os.path.join(video_frame_root, vid)
    mel_path = os.path.join(audio_mel_root, f"{vid}.npy")

    if not os.path.isdir(frame_dir):
        print(f"Skipping {vid}: frame directory missing.")
        return
    if not os.path.isfile(mel_path):
        print(f"Skipping {vid}: mel file missing.")
        return

    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))])
    frames = [transform(Image.open(fp).convert('RGB')) for fp in frame_paths]
    frames = torch.stack(frames).unsqueeze(0).to(device)

    mel = torch.from_numpy(np.load(mel_path)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        frame_feature = model.encode_video(frames, normalize=True, pool=False)

    os.makedirs(save_feature_root, exist_ok=True)
    torch.save({
        'frame_feature': frame_feature.cpu(),
        'mel_feature': mel.cpu()
    }, os.path.join(save_feature_root, f"{vid}_feature.pt"))

    print(f"Saved features for {vid}.")

def extract_features_sequential(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_stage1_model(args.cavp_config_path, args.cavp_ckpt_path, device)

    video_ids = sorted([
        vid for vid in os.listdir(args.video_frame_root)
        if os.path.isdir(os.path.join(args.video_frame_root, vid))
    ])

    start = args.start_vid if args.start_vid is not None else 0
    end = args.end_vid if args.end_vid is not None else len(video_ids)
    selected_videos = video_ids[start:end]

    for vid in tqdm(selected_videos, desc=f"Extracting features [{start}:{end}]"):
        extract_one_video(vid, args.video_frame_root, args.audio_mel_root, args.save_feature_root, model, device)

    print("\nFeature extraction done for selected range.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_vid", type=int, default=None)
    parser.add_argument("--end_vid", type=int, default=None)
    parser.add_argument("--video_frame_root", type=str, required=True)
    parser.add_argument("--audio_mel_root", type=str, required=True)
    parser.add_argument("--save_feature_root", type=str, required=True)
    parser.add_argument("--cavp_config_path", type=str, default="inference/config/Stage1_CAVP.yaml")
    parser.add_argument("--cavp_ckpt_path", type=str, default="inference/diff_foley_ckpt/cavp_epoch66.ckpt")
    args = parser.parse_args()
    extract_features_sequential(args)
