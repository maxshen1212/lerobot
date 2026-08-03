# Graphen — Bimanual SO-101 real-robot cheatsheet

本專案的實機命令記錄。目前 real 資料集：
`ChihHanShen/bimanual-so101-pickvials-real` @ `/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real`

## 0. 一次性設定 (setup)

```bash
lerobot-find-port                     # 找 leader/follower 序列埠
graphen-setup-udev                    # 建立 /dev/tty{Leader,Follower}{Left,Right} 穩定 symlink
lerobot-find-cameras realsense        # 列出 RealSense 序號 (貼進 record/teleop config)

lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml
lerobot-calibrate --config_path=calibration/config/bimanual_so101_leader_config.yaml
```

## 1. Teleop (無錄製，純遙操作測試)

```bash
lerobot-teleoperate --config_path=calibration/config/bimanual_so101_teleoperate_config.yaml
```

## 2. 錄製資料集 (record)

```bash
lerobot-record --config_path=calibration/config/bimanual_so101_record_config.yaml
```

錄製中鍵盤控制 (pynput 全域監聽，需有畫面；無頭環境停用，擷取按鍵可能需 sudo)：

| 鍵         | 錄製中               | 重置等待中               |
| ---------- | -------------------- | ------------------------ |
| → 右方向鍵 | 停止這一集、進下一步 | 略過等待、直接開始下一集 |
| ← 左方向鍵 | 丟棄並重錄上一集     | —                        |
| Esc        | 完全停止並存檔       | 完全停止並存檔           |

> 編碼參數 (streaming / encoder*threads=2 / queue=90 / av1) 已在 record config 內調校，
> 與 sim 資料集 codec 對齊。相機 key = `wrist_left` / `center` / `wrist_right` (無 left*/right\_ 前綴)。

## 3. 檢查資料品質 (QA)

```bash
# (a) 影片幀數 vs action/state 幀數是否對齊 (抓 streaming 掉幀)
uv run python tools/check_frame_alignment.py datasets/bimanual-so101-pickvials-real

# (b) 列出每個 episode 對應的 mp4 檔號與時間段 (一個 mp4 內含多集!)
uv run python tools/view_episode.py datasets/bimanual-so101-pickvials-real --list

# (c) 抽出單一 episode、三台相機一起切片並開啟 (例: ep 47)
uv run python tools/view_episode.py datasets/bimanual-so101-pickvials-real 47
#   單一相機、不自動開: --cam center --no-open

# (d) 完整視覺化: 影片 + 12 維 state/action 曲線同步播放 (Rerun 視窗)
uv run lerobot-dataset-viz \
  --repo-id ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real --episode-index 47
```

> ⚠️ mp4 檔名是「檔號 file-XXX」，**不是 episode_index**，且一個 mp4 串了多集。
> 想看某一集一定要用 `view_episode.py` 或 `dataset-viz` 查表，不能直接點 mp4 猜。

## 4. 刪除品質不好的 episode

```bash
uv run lerobot-edit-dataset \
  --repo_id ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real \
  --new_root datasets/bimanual-so101-pickvials-real-clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[24, 43]"

uv run lerobot-edit-dataset \
  --repo_id ChihHanShen/bimanual-so101-pickvials-real-15fps \
  --root datasets/bimanual-so101-pickvials-real-15fps \
  --new_root datasets/bimanual-so101-pickvials-real-15fps-clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[45, 46, 47, 48]"

# 刪完必驗
uv run python tools/check_frame_alignment.py datasets/bimanual-so101-pickvials-real
```

> **兩個坑：**
>
> 1. **一定要給 `--root`** (v2 語意：`--root`/`--new_root` = 資料集資料夾本身，不再拼 repo_id)。
>    不給會去找 HF 快取而找不到。`--root`==`--new_root` = 原地改並自動留 `_bak` 備份。
> 2. **`episode_indices` 用 episode_index，每刪一次就 reindex** → 要刪的一次全列進去，勿分批。
>
> (workshop 的 `tools/delete_episodes.py` 是用穩定「檔號」刪、且 ROOT 寫死指向 sim 資料集，勿混用。)

## 5. Replay (在真機重播某一集，驗證校正/接線)

```bash
lerobot-replay \
  --robot.type=bi_so_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/graphen/sim2real/lerobot/calibration/bimanual_follower \
  --robot.left_arm_config.port=/dev/ttyFollowerLeft  --robot.left_arm_config.use_degrees=true \
  --robot.right_arm_config.port=/dev/ttyFollowerRight --robot.right_arm_config.use_degrees=true \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real \
  --dataset.root=/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real\
  --dataset.episode=0 \
  --dataset.fps=30

lerobot-replay \
  --robot.type=bi_so_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/graphen/sim2real/lerobot/calibration/bimanual_follower \
  --robot.left_arm_config.port=/dev/ttyFollowerLeft  --robot.left_arm_config.use_degrees=true \
  --robot.right_arm_config.port=/dev/ttyFollowerRight --robot.right_arm_config.use_degrees=true \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real-15fps \
  --dataset.root=/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real-15fps \
  --dataset.episode=44 \
  --dataset.fps=15


```

## 6. Hugging Face 下載 / 上傳

這版 lerobot 沒有專門的上傳/下載 CLI；下載靠 `hf download` (huggingface_hub CLI)，
上傳靠 Python `push_to_hub()`。
本機資料夾名 (底線) 與 HF 目標名 (連字號) 不同 — repo_id 只是上傳目標標籤，info.json 不存它，換名上傳沒問題。

```bash
# 一次性登入 (下載 private repo / 上傳都需要，寫入需 write token: https://huggingface.co/settings/tokens)
uv run hf auth login
```

### 6a. 下載資料集 (download)

```bash
cd /home/graphen/sim2real/lerobot

# --repo-type dataset 必填 (預設是 model)；--local-dir 直接落地成平舖資料夾，
# 符合本專案 datasets/<name> 的慣例 (不要用預設 HF cache，list_episodes.py / delete_episodes.py 認的是這個路徑)
uv run hf download ChihHanShen/bimanual-so101-pickvials-real-V2 \
  --repo-type dataset \
  --local-dir datasets/bimanual-so101-pickvials-real-V2

uv run hf download ChihHanShen/bimanual-so101-pickvials-real-15fps \
  --repo-type dataset \
  --local-dir datasets/bimanual-so101-pickvials-real-15fps
```

> 資料夾若已存在，`hf download` 只會補缺少/變動的檔 (斷點續傳)，不會整包重下。

### 6b. 首次上傳 / 只是加更多集 (append)

```bash
cd /home/graphen/sim2real/lerobot
uv run python -c '
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    "ChihHanShen/bimanual-so101-pickvials-real",          # HF 目標名 (連字號)
    root="datasets/bimanual-so101-pickvials-real",   # 本機資料夾 (底線)
)
ds.push_to_hub(private=False, tags=["so101","bimanual","real","lerobot"])
'
```

> `push_to_hub()` 是「覆蓋 + 新增」，**不會刪掉 Hub 上本機已無的檔**。只加集時完全正確。
> 自動上傳 meta/ data/ videos/ (跳過 images/)、建 repo、產 dataset card、打版本 tag。
> 資料已由 lerobot-record 收尾過，無需再 finalize()。訓練時要用 HF 上的**連字號**名稱。

### 6c. 刪過集之後重新上傳 (鏡像，清掉孤兒檔) ⚠️

刪集會重新打包 shard、檔案數可能變少 → 單純重推會在 Hub 留下沒人引用的孤兒 mp4。
用 `delete_patterns` 讓 Hub 精確鏡像本機 (README 卡片不動)：

```bash
cd /home/graphen/sim2real/lerobot
uv run python -c '
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
ds = LeRobotDataset(
    "ChihHanShen/bimanual-so101-pickvials-real-15fps",
    root="/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real-15fps",
)
HfApi().upload_folder(
    repo_id=ds.repo_id, repo_type="dataset", folder_path=str(ds.root),
    ignore_patterns=["images/"],
    delete_patterns=["data/*", "videos/*", "meta/*"],   # 刪掉本機已無的舊 shard
)
'
```

> 替代法：直接在 HF 網頁刪掉整個 repo，再跑 6a 從頭推一份 (最乾淨，代價是失去 likes/歷史)。
