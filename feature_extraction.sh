#!/usr/bin/bash

#SBATCH -J CAVPMask
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 3-0
#SBATCH -o /data/jhlee39/workspace/logs/slurm-%A.out

# 직접 설정: 비디오 범위
START_VID=
END_VID=

# 직접 설정: 경로
VIDEO_FRAME_ROOT="dataset/frames/frame_mask/black_random8/testset" # black_last16  black_last31  black_last4  black_last8  black_random4  black_random8  lowres_all  scatter_all  superres_all  vertical_all
AUDIO_MEL_ROOT="dataset/mel/testset"
SAVE_FEATURE_ROOT="dataset/features/feature_mask/black_random8/testset"

# 선택 사항: 설정 파일 경로
CAVP_CONFIG="inference/config/Stage1_CAVP.yaml"
CAVP_CKPT="inference/diff_foley_ckpt/cavp_epoch66.ckpt"

# 인자 구성
ARGS="--video_frame_root $VIDEO_FRAME_ROOT --audio_mel_root $AUDIO_MEL_ROOT --save_feature_root $SAVE_FEATURE_ROOT"
ARGS="$ARGS --cavp_config_path $CAVP_CONFIG --cavp_ckpt_path $CAVP_CKPT"

if [ ! -z "$START_VID" ]; then
  ARGS="$ARGS --start_vid $START_VID"
fi
if [ ! -z "$END_VID" ]; then
  ARGS="$ARGS --end_vid $END_VID"
fi

python feature_extraction.py $ARGS
