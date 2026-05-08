```bash
# conda
conda create -y -n lerobot python=3.12
conda activate lerobot
# must install this
conda install ffmpeg=7.1.1 -c conda-forge
cd lerobot && pip install -e .
conda env list
conda deactivate
conda remove -y -n lerobot --all

pip install lerobot[multi_task_dit]

pip install -e ".[libero]"

# LIBERO test
sed -i 's|/tmp/robosuite.log|/data/maxshen/robosuite.log|' \
  /data/maxshen/miniconda3/envs/lerobot/lib/python3.12/site-packages/robosuite/utils/log_utils.py

python -m robosuite.scripts.setup_macros

python -c "from robosuite.macros_private import *; print('OK')"


# tmux
tmux new -s window_name
crrl + b, d
tmux kill-session -t window_name
tmux ls
tmux a -t window_name

# utilities
ps -fp "<pid>"
quota -s
du -sh ~/.cache/* 2> /dev/null | sort -h | tail -n 30
rm -rf ~/.cache/wandb/artifacts
rsync -avzP --progress a b
watch -n 1 nvidia-smi
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES
export MUJOCO_GL=egl

# lerobot training
conda activate lerobot
cd data/maxshen/lerobot
wandb login
hf auth login

#########################################
#    Official Lerobot Training Script   #
#########################################

# Diffusion Policy (training all episodes)
lerobot-train \
  --dataset.repo_id=lbm_sim/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
  --dataset.root=/data/maxshen/lerobot_training_data_v3/ego_BimanualPlaceAppleFromBowlOnCuttingBoard_after_split/train \
  --policy.repo_id=CHIH-HAN/tri-diffusion-BimanualPlaceAppleFromBowlOnCuttingBoard \
  --val_root=/data/maxshen/lerobot_training_data_v3/ego_BimanualPlaceAppleFromBowlOnCuttingBoard_after_split/val \
  --batch_size=64 \
  --steps=80000 \
  --log_freq=100 \
  --save_freq=5000 \
  --val_freq=5000 \
  --num_workers=32 \
  --save_checkpoint=true \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --policy.pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" \
  --policy.use_group_norm=false \
  --policy.resize_shape=[240,320] \
  --policy.spatial_softmax_num_keypoints=64 \
  --policy.optimizer_lr=5e-5 \ # not yet added and test
  --policy.crop_ratio=0.9 \
  --policy.n_action_steps=4 \
  --policy.noise_scheduler_type=DDIM \
  --wandb.enable=true \
  --wandb.entity=chihhans-usc \
  --wandb.project=SLURM_Lerobot \
  --dataset.image_transforms.enable=true

# DiT (training all episodes)
export CUDA_VISIBLE_DEVICES=5,6,7
echo $CUDA_VISIBLE_DEVICES


accelerate launch \
  --multi_gpu \
  --num_processes=3 \
  --main_process_port 0 \
  $(which lerobot-train) \
  --dataset.repo_id=lbm_sim/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
  --dataset.root=/data/maxshen/lerobot_training_data_v3/ego_BimanualPlaceAppleFromBowlOnCuttingBoard_lang_after_split/train \
  --val_root=/data/maxshen/lerobot_training_data_v3/ego_BimanualPlaceAppleFromBowlOnCuttingBoard_lang_after_split/val \
  --policy.type=multi_task_dit \
  --policy.repo_id=CHIH-HAN/tri-DiT-BimanualPlaceAppleFromBowlOnCuttingBoard \
  --policy.push_to_hub=true \
  --job_name="multitask-dit" \
  --wandb.enable=true \
  --wandb.entity=chihhans-usc \
  --wandb.project=SLURM_Lerobot \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs='{"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}' \
  --dataset.video_backend=torchcodec \
  --policy.use_amp=true \
  --policy.noise_scheduler_type=DDIM \
  --policy.train_diffusion_n_samples=8 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.n_obs_steps=2 \
  --policy.num_inference_steps=8 \
  --policy.use_rope=false \
  --policy.use_positional_encoding=true \
  --policy.hidden_dim=768 \
  --policy.num_layers=10 \
  --policy.num_heads=12 \
  --policy.dropout=0.1 \
  --policy.timestep_embed_dim=256 \
  --policy.objective=diffusion \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-6 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape=[256,342] \
  --policy.image_crop_shape=[224,224] \
  --policy.image_crop_is_random=true \
  --policy.text_encoder_name=openai/clip-vit-base-patch32 \
  --policy.vision_encoder_lr_multiplier=0.1 \
  --policy.scheduler_name=constant \
  --policy.device=cuda \
  --num_workers=14 \
  --save_freq=5000 \
  --log_freq=100 \
  --val_freq=5000 \
  --steps=50000 \
  --batch_size=32 \
  --eval_freq=5000000

#########################################
#         LIBERO Training Script        #
#########################################
accelerate launch \
  --multi_gpu \
  --num_processes=3 \
  --main_process_port 0 \
  $(which lerobot-train) \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --output_dir="./outputs/multitask_dit_libero" \
  --policy.type=multi_task_dit \
  --policy.push_to_hub=false \
  --job_name="multitask-dit-libero" \
  --wandb.enable=true \
  --wandb.entity=chihhans-usc \
  --wandb.project=SLURM_Lerobot \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs='{"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}' \
  --dataset.video_backend=torchcodec \
  --policy.use_amp=true \
  --policy.noise_scheduler_type=DDIM \
  --policy.train_diffusion_n_samples=8 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.n_obs_steps=2 \
  --policy.num_inference_steps=8 \
  --policy.use_rope=false \
  --policy.use_positional_encoding=true \
  --policy.hidden_dim=768 \
  --policy.num_layers=10 \
  --policy.num_heads=12 \
  --policy.dropout=0.1 \
  --policy.timestep_embed_dim=256 \
  --policy.objective=diffusion \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-6 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape=[256,342] \
  --policy.image_crop_shape=[224,224] \
  --policy.image_crop_is_random=true \
  --policy.text_encoder_name=openai/clip-vit-base-patch32 \
  --policy.vision_encoder_lr_multiplier=0.1 \
  --policy.scheduler_name=constant \
  --policy.device=cuda \
  --num_workers=14 \
  --save_freq=5000 \
  --log_freq=100 \
  --steps=50000 \
  --batch_size=32 \
  --eval_freq=5000000

  python /data/maxshen/lerobot/run_eval.py \
  --output_dir=./eval_logs/ \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/data/maxshen/lerobot/outputs/multitask_dit_libero/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=8 \
  --env.max_parallel_tasks=2

  python /data/maxshen/lerobot/run_eval.py \
  --output_dir=./eval_logs/ \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/data/maxshen/lerobot/outputs/multitask_dit_libero/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=8 \
  --env.max_parallel_tasks=5


#########################################
#  Masquerade's Lerobot Training Script #
#########################################

# Author's training script example
python lerobot/scripts/train.py \
  --dataset.repo_id="[stack_pots_240, epic_kitchens_v1]" \
  --policy.type=diffusion \
  --output_dir=outputs/train/stack_pots_240_epic_v20 \
  --job_name stack_pots_240_v20 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.use_auxiliary_mlp=true \
  --policy.pretrained_backbone_weights=/home/masquerade/outputs/v20/snapshot.pt \
  --policy.use_film_cond=true \
  --policy.cotrain_debug=true \
  --policy.cotrain_debug_freq=1000 \
  --batch_size=64 \
  --policy.auxiliary_loss_weight=10.0 --seed 2
```
