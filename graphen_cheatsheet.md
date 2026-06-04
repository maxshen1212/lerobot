# official example: single so-101 calibration

lerobot-calibrate \
 --robot.type=so101_follower \
 --robot.port=/dev/lerobot_follower_right \
 --robot.id=my_awesome_follower_arm

# Setting steps

graphen-setup-udev

lerobot-find-port

// list connected RealSense cameras + their serial numbers (paste into the teleop config)

lerobot-find-cameras realsense

lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml

lerobot-calibrate --config_path=calibration/config/bimanual_so101_leader_config.yaml

lerobot-teleoperate --config_path=calibration/config/bimanual_so101_teleoperate_config.yaml

# record a dataset (RIGHT=next episode, LEFT=re-record, ESC=stop)

lerobot-record --config_path=calibration/config/bimanual_so101_record_config.yaml

lerobot-replay \
 --robot.type=bi_so_follower \
 --robot.id=bimanual_so101_follower \
 --robot.calibration_dir=/home/max/Desktop/lerobot/calibration/bimanual_follower \
 --robot.left_arm_config.port=/dev/lerobot_follower_left \
 --robot.left_arm_config.use_degrees=true \
 --robot.right_arm_config.port=/dev/lerobot_follower_right \
 --robot.right_arm_config.use_degrees=true \
 --dataset.repo_id=ChihHanShen/bimanual_so101_pickplace_20260602_161251 \
 --dataset.root=/home/max/Desktop/lerobot/datasets/bimanual_so101_pickplace \
 --dataset.episode=0 \
 --dataset.fps=30

lerobot-edit-dataset \
    --repo_id ChihHanShen/bimanual_so101_pickplace \
    --operation.type delete_episodes \
    --operation.episode_indices "[4, 21, 22, 64]"

# ── train a diffusion policy (the run that produced the checkpoint below) ──
lerobot-train \
  --dataset.repo_id=ChihHanShen/bimanual_so101_pickplace_95 \
  --dataset.root=/home/max/Desktop/lerobot/datasets/bimanual_so101_pickplace_95 \
  --policy.repo_id=CHIH-HAN/graphen-diffusion \
  --batch_size=4 --steps=1000 --log_freq=100 --save_freq=5000 --num_workers=8 \
  --save_checkpoint=true \
  --policy.type=diffusion --policy.device=cuda \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --policy.pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" \
  --policy.use_group_norm=false \
  --policy.resize_shape=[240,320] --policy.crop_ratio=0.9 \
  --policy.noise_scheduler_type=DDIM \
  --wandb.enable=false --wandb.entity=chihhans-usc --wandb.project=Graphen \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false

# ── inference / deploy the trained policy on the real robot ──
# lerobot-record is data-collection only now; use lerobot-rollout to run a policy.
# --strategy.type=base  -> autonomous rollout, no recording.
lerobot-rollout \
  --strategy.type=base \
  --policy.path=/home/max/Desktop/lerobot/outputs/train/2026-06-04/16-23-37_diffusion/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --robot.type=bi_so_follower \
  --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/max/Desktop/lerobot/calibration/bimanual_follower \
  --robot.left_arm_config.port=/dev/lerobot_follower_left \
  --robot.left_arm_config.use_degrees=true \
  --robot.left_arm_config.cameras='{ego_centric: {type: intelrealsense, serial_number_or_name: "138422072598", width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.port=/dev/lerobot_follower_right \
  --robot.right_arm_config.use_degrees=true \
  --robot.right_arm_config.cameras='{third_person: {type: intelrealsense, serial_number_or_name: "215322078630", width: 640, height: 480, fps: 30}}' \
  --task="Pick up the cube and place it in the box." \
  --duration=60

# Optional: record evaluation episodes while the policy runs (auto-upload off):
#   swap  --strategy.type=base  for:
#     --strategy.type=sentry \
#     --dataset.repo_id=ChihHanShen/bimanual_so101_pickplace_eval \
#     --dataset.root=/home/max/Desktop/lerobot/datasets/bimanual_so101_pickplace_eval \
#     --dataset.single_task="Pick up the cube and place it in the box."
