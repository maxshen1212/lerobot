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
