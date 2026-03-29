```bash
# conda
conda create -y -n lerobot python=3.12
# must install this
conda install ffmpeg=7.1.1 -c conda-forge
conda activate lerobot
conda env list
conda deactivate
conda remove -y -n lerobot --all

# tmux
tmux new -s window_name
crrl + b, d
tmux kill-session -t window_name
tmux ls
tmux a -t window_name

# utilities
rsync -av --progress a b
watch -n 1 nvidia-smi
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES

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
