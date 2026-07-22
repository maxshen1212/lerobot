# Setting steps

lerobot-find-port
graphen-setup-udev

// list connected RealSense cameras + their serial numbers (paste into the teleop config)

lerobot-find-cameras realsense

lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml

lerobot-calibrate --config_path=calibration/config/bimanual_so101_leader_config.yaml

lerobot-teleoperate --config_path=calibration/config/bimanual_so101_teleoperate_config.yaml

# 錄製資料集 (record a dataset)

#

# 鍵盤控制 (pynput 全域監聽；需要有畫面顯示 — 無頭環境下會停用):

# → 右方向鍵 : 提早結束目前步驟。錄製中 = 停止錄製這一集並進入下一步；

# 重置等待期間 = 略過等待，直接開始下一集。

# ← 左方向鍵 : 提早結束並「重新錄製」上一集 (丟棄後重錄)。

# Esc : 完全停止資料錄製 (結束整個錄製流程)。

#

# 注意：擷取按鍵可能需要 sudo，讓終端機能夠監聽鍵盤事件。

lerobot-record --config_path=calibration/config/bimanual_so101_record_config.yaml

lerobot-replay \
 --robot.type=bi_so_follower \
 --robot.id=bimanual_so101_follower \
 --robot.calibration_dir=/home/graphen/sim2real/lerobot/calibration/bimanual_follower \
 --robot.left_arm_config.port=/dev/ttyFollowerLeft \
 --robot.left_arm_config.use_degrees=true \
 --robot.right_arm_config.port=/dev/ttyFollowerRight \
 --robot.right_arm_config.use_degrees=true \
 --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials \
 --dataset.root=/home/graphen/sim2real/lerobot/datasets/bimanual_so101_vial_pickplace_real \
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

# --strategy.type=base -> autonomous rollout, no recording.

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

# swap --strategy.type=base for:

# --strategy.type=sentry \

# --dataset.repo_id=ChihHanShen/bimanual_so101_pickplace_eval \

# --dataset.root=/home/max/Desktop/lerobot/datasets/bimanual_so101_pickplace_eval \

# --dataset.single_task="Pick up the cube and place it in the box."
