# Graphen — Bimanual SO-101 real-robot cheatsheet

真機命令記錄。目前資料集 `ChihHanShen/bimanual-so101-pickvials-real`
@ `datasets/bimanual-so101-pickvials-real`。

> **環境：`source ~/env_isaaclab/bin/activate`，指令直接呼叫、不要加 `uv run`。**
> 這個 checkout（`n1.7-graphen`，v0.4.3）與 sim 共用同一個 venv，本目錄**沒有自己的 `.venv`**
> —— 在這裡跑 `uv run` 會就地生一個出來。

---

## 0. 一次性設定

```bash
graphen-setup-udev --apply      # 建立 /dev/tty{Leader,Follower}{Left,Right}（換 USB 轉板才要 --identify）
lerobot-find-cameras realsense  # 列出 RealSense 序號，貼進 record config

lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml
lerobot-calibrate --config_path=calibration/config/bimanual_so101_leader_config.yaml

git add calibration/ && git commit -m "calib: baseline for <dataset name>"
```

> **收資料與 eval 必須是同一組 calibration**，否則 policy 看到的正規化分布不一樣。
> 沒事**不要**重跑 `lerobot-calibrate`。完整流程見 **[CALIBRATION.md](CALIBRATION.md)**。

## 0.5 每次開工前

```bash
graphen-setup-udev              # symlink 是否都在、序號是否正確
git status calibration/         # 應該乾淨；有 M → git checkout calibration/
```

維修過手臂要另外比對硬限位，見 [CALIBRATION.md](CALIBRATION.md) §4。

---

## 1. Teleop（純遙操作，不錄製）

```bash
lerobot-teleoperate --config_path=calibration/config/bimanual_so101_teleoperate_config.yaml
```

## 2. 錄製資料集

```bash
lerobot-record --config_path=calibration/config/bimanual_so101_record_config.yaml
```

錄製中鍵盤控制（pynput 全域監聽，需有畫面；無頭環境停用）：

| 鍵 | 錄製中 | 重置等待中 |
| --- | --- | --- |
| → | 停止這一集、進下一步 | 略過等待、直接開始下一集 |
| ← | 丟棄並重錄上一集 | — |
| Esc | 完全停止並存檔 | 完全停止並存檔 |

> **第一次錄要盯 fps 穩不穩。** 0.4.x 是 writer thread 寫 PNG、再於 `save_episode()` 編碼
> （config 內已設 `num_image_writer_processes/threads_per_camera/video_encoding_batch_size`
> = `0 / 4 / 1`）。不穩就把 `num_image_writer_processes` 調到 1 以上。

## 3. 檢查資料品質

```bash
lerobot-dataset-viz \
  --repo-id ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real --episode-index 47
```

> ⚠️ mp4 檔名是「檔號 file-XXX」，**不是 episode_index**，且一個 mp4 串了多集
> —— 要用 `dataset-viz` 查表，不能直接點 mp4 猜。

## 4. 刪除品質不好的 episode

```bash
lerobot-edit-dataset \
  --repo_id ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real \
  --new_root datasets/bimanual-so101-pickvials-real-clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[24, 43]"
```

> 1. **一定要給 `--root`**（= 資料集資料夾本身，不再拼 repo_id）。不給會去找 HF 快取而找不到。
>    `--root` == `--new_root` = 原地改並自動留 `_bak`。
> 2. **每刪一次就 reindex** → 要刪的一次全列進去，勿分批。
>
> workshop 的 `tools/delete_episodes.py` 是用穩定「檔號」刪、ROOT 寫死指向 sim 資料集，**勿混用**。

## 5. Replay（在真機重播某一集，驗證校正/接線）

```bash
lerobot-replay \
  --robot.type=bi_so101_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/graphen/sim2real/lerobot/calibration/bimanual_follower \
  --robot.left_arm_port=/dev/ttyFollowerLeft \
  --robot.right_arm_port=/dev/ttyFollowerRight \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real \
  --dataset.root=/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real \
  --dataset.episode=0 \
  --dataset.fps=30
```

> **`--dataset.fps` 要對上該資料集自己的 fps**（看 `meta/info.json`），不是照抄 30。

---

## 6. Hugging Face 下載 / 上傳

沒有專門的 CLI：下載用 `hf download`，上傳用 Python `push_to_hub()`。
本機資料夾名（底線）與 HF 名（連字號）不同沒關係，`info.json` 不存 repo_id。

```bash
hf auth login       # 一次性；上傳需 write token
```

**下載** —— `--repo-type dataset` 必填（預設是 model）；`--local-dir` 才會落地成平舖資料夾：

```bash
hf download ChihHanShen/bimanual-so101-pickvials-real \
  --repo-type dataset --local-dir datasets/bimanual-so101-pickvials-real
```

**上傳（首次 / 只是加更多集）** —— `push_to_hub()` 是「覆蓋 + 新增」，只加集時完全正確：

```bash
python -c '
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("ChihHanShen/bimanual-so101-pickvials-real",
                    root="datasets/bimanual-so101-pickvials-real")
ds.push_to_hub(private=False, tags=["so101","bimanual","real","lerobot"])
'
```

**⚠️ 刪過集之後重新上傳** —— 刪集會重新打包 shard，單純重推會在 Hub 留下孤兒 mp4。
用 `delete_patterns` 讓 Hub 精確鏡像本機（README 卡片不動）：

```bash
python -c '
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
ds = LeRobotDataset("ChihHanShen/bimanual-so101-pickvials-real",
                    root="/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real")
HfApi().upload_folder(
    repo_id=ds.repo_id, repo_type="dataset", folder_path=str(ds.root),
    ignore_patterns=["images/"],
    delete_patterns=["data/*", "videos/*", "meta/*"],   # 刪掉本機已無的舊 shard
)
'
```

---

## 從 v0.6.1 過來要注意的差異

| | v0.6.1 | 現在（v0.4.3） |
| --- | --- | --- |
| 型別 | `bi_so_follower` / `bi_so_leader` | **`bi_so101_follower` / `bi_so101_leader`**（本 repo 新增） |
| 每臂欄位 | `--robot.left_arm_config.port` | **`--robot.left_arm_port`**（扁平） |
| 正規化 | `use_degrees=true`（度） | **不設**（預設 `False` = ±100） |
| 跑 policy | `lerobot-rollout` | **不存在** → `lerobot-record --policy.path=...` |
| diffusion 裁切 | `resize_shape` + `crop_ratio` | `resize_shape` **不存在**；改 `--policy.crop_shape=[H,W]`<br>⚠️ 預設 `(84,84)` 是 PushT 用的，對 480×640 會裁爛且不報錯 |
| `tools/*.py` | 有 | **沒遷過來**，這個 checkout 沒有 `tools/` |

**舊真機資料集是「度」錄的，和現在的 ±100 不可混用**，要重收。
