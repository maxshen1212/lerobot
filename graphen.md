```bash
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES
lerobot-train \
  --dataset.repo_id=ChihHanShen/bimanual_so101_pickplace \
  --policy.repo_id=CHIH-HAN/tri-diffusion-BimanualPlaceAppleFromBowlOnCuttingBoard \
  --batch_size=64 \
  --steps=300000 \
  --log_freq=100 \
  --save_freq=5000 \
  --num_workers=32 \
  --save_checkpoint=true \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --policy.pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" \
  --policy.use_group_norm=false \
  --policy.resize_shape=[240,320] \
  --policy.crop_ratio=0.9 \
  --policy.noise_scheduler_type=DDIM \
  --wandb.enable=true \
  --wandb.entity=chihhans-usc \
  --wandb.project=Graphen \
  --dataset.image_transforms.enable=true
