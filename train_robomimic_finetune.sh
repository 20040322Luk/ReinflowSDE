#!/bin/bash

# Robomimic Fine-tuning Script for ReinFlow (Image-based)
# This script fine-tunes a ReinFlow policy on the Robomimic Square environment using PPO and image observations.

# Configuration:
# --config-dir: Path to the configuration directory.
# --config-name: 'ft_ppo_reflow_mlp_img' for image-based PPO fine-tuning of ReinFlow.

echo "Starting Robomimic Fine-tuning..."

# 设置环境变量以强制无头渲染，避免EGL相关错误
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY

python script/run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_reflow_mlp_img \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
    device=cuda:0 \
    sim_device=null \
    +env.headless=true

echo "Fine-tuning finished."