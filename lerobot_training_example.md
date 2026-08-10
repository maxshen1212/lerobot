```bash
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES
lerobot-train \
  --dataset.repo_id=ChihHanShen/bimanual_so101_pickplace_95 \
  --dataset.root=/data/maxshen/lerobot/datasets/bimanual_so101_pickplace_95 \
  --policy.repo_id=CHIH-HAN/graphen-diffusion \
  --batch_size=64 \
  --steps=200000 \
  --log_freq=100 \
  --save_freq=5000 \
  --num_workers=8 \
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
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=true
```

`lerobot-record` 只做資料收集；跑 policy 用 `lerobot-rollout`。`--strategy.type=base` = 純自主 rollout、不錄製。

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=/home/max/Desktop/lerobot/outputs/train/2026-06-04/16-23-37_diffusion/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --robot.type=bi_so_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/max/Desktop/lerobot/calibration/bimanual_follower \
  --robot.left_arm_config.port=/dev/lerobot_follower_left  --robot.left_arm_config.use_degrees=true \
  --robot.left_arm_config.cameras='{ego_centric: {type: intelrealsense, serial_number_or_name: "138422072598", width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.port=/dev/lerobot_follower_right --robot.right_arm_config.use_degrees=true \
  --robot.right_arm_config.cameras='{third_person: {type: intelrealsense, serial_number_or_name: "215322078630", width: 640, height: 480, fps: 30}}' \
  --task="Pick up the cube and place it in the box." --duration=60
```

> 想在 rollout 同時錄評估集：把 `--strategy.type=base` 換成 `--strategy.type=sentry`，
> 並加 `--dataset.repo_id=..._eval --dataset.root=... --dataset.single_task="..."` (auto-upload 預設關)。
