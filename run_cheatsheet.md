# Graphen — Bimanual SO-101 real-robot cheatsheet

本專案的實機命令記錄。目前 real 資料集：
`ChihHanShen/bimanual-so101-pickvials-real` @ `/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real`

> **環境：`source ~/env_isaaclab/bin/activate`。**
> 這個 checkout（branch `n1.7-graphen`，v0.4.3）與 sim 端共用同一個 venv，
> **本目錄下沒有自己的 `.venv`**。所以下面的指令一律直接呼叫，
> **不要加 `uv run`** —— 在這裡跑 `uv run` 會就地生出一個新的 `.venv`，
> 那正是這次遷移要消掉的東西。

## 0. 一次性設定 (setup)

```bash
graphen-setup-udev --apply            # 依 repo 內的序號表建立 /dev/tty{Leader,Follower}{Left,Right}
                                      # (只有換 USB 轉板才需要 --identify 重新辨識序號)
lerobot-find-cameras realsense        # 列出 RealSense 序號 (貼進 record/teleop config)

lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml
lerobot-calibrate --config_path=calibration/config/bimanual_so101_leader_config.yaml

# 校準完立刻 commit，這個 commit 就是這批 dataset 的基準
git add calibration/ && git commit -m "calib: baseline for <dataset name>"
```

> **Calibration 一致性是這個專案的硬性前提** — 收資料與 eval 必須是同一組 calibration，
> 否則 policy 看到的正規化分布會不一樣。完整流程（每次開工的檢查、維修後怎麼判斷有沒有
> 跑掉、跑掉了怎麼救）見 **[CALIBRATION.md](CALIBRATION.md)**。
> 沒事**不要**重跑 `lerobot-calibrate`。
>
> 這個版本用 **RANGE_M100_100（±100）**，`use_degrees` 一律不設。
> 在 ±100 下 span 直接就是尺度，**每個關節都必須掃到真正的硬限位**。
>
> 四個 YAML 的型別是 **`bi_so101_follower` / `bi_so101_leader`**（本 repo 自己加的，
> 見 CALIBRATION.md §1.2）。**不要改回 `bi_so100_*`** —— 那條路徑會把 `wrist_roll`
> 寫死成 0–4095。

## 0.5 每次開工前的檢查

```bash
graphen-setup-udev              # USB symlink 是否都在且序號正確
git status calibration/         # 應該是乾淨的；有 M 就代表被重新校準過 → git checkout calibration/
```

維修過手臂之後，要另外掃硬限位跟基準比對，見 [CALIBRATION.md](CALIBRATION.md) §4。

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

> **編碼在 0.4.x 是另一套模型。** 0.6.x 的 streaming 編碼選項
> (`streaming_encoding` / `encoder_queue_maxsize` / `encoder_threads`) **不存在**；
> 0.4.x 是先用 writer thread 寫 PNG，再在 `save_episode()` 內編碼。
> 對應參數是 `num_image_writer_processes` / `num_image_writer_threads_per_camera`
> / `video_encoding_batch_size`，已在 record config 內設成 `0 / 4 / 1`。
> **第一次錄要留意 fps 穩不穩**，不穩就把 `num_image_writer_processes` 調到 1 以上。
>
> 相機 key = `wrist_left` / `center` / `wrist_right` (無 left*/right\_ 前綴) —— 0.4.x 原生就是這樣，
> 不需要 0.6.x 那個 workaround。

## 3. 檢查資料品質 (QA)

```bash
# 影片 + 12 維 state/action 曲線同步播放 (Rerun 視窗)
lerobot-dataset-viz \
  --repo-id ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real --episode-index 47
```

> ⚠️ mp4 檔名是「檔號 file-XXX」，**不是 episode_index**，且一個 mp4 串了多集。
> 想看某一集要用 `dataset-viz` 查表，不能直接點 mp4 猜。
>
> **`tools/check_frame_alignment.py` 與 `tools/view_episode.py` 沒有遷過來**
> （從 0.6.1 遷過來時判定不需要），這個 checkout 沒有 `tools/`。

## 4. 刪除品質不好的 episode

```bash
lerobot-edit-dataset \
  --repo_id ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real \
  --new_root datasets/bimanual-so101-pickvials-real-clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[24, 43]"

lerobot-edit-dataset \
  --repo_id ChihHanShen/bimanual-so101-pickvials-real-15fps \
  --root datasets/bimanual-so101-pickvials-real-15fps \
  --new_root datasets/bimanual-so101-pickvials-real-15fps-clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[45, 46, 47, 48]"


# (刪完的驗證: 用 lerobot-info / lerobot-dataset-viz 抽查, tools/ 沒有遷過來)
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
  --robot.type=bi_so101_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/graphen/sim2real/lerobot/calibration/bimanual_follower \
  --robot.left_arm_port=/dev/ttyFollowerLeft \
  --robot.right_arm_port=/dev/ttyFollowerRight \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real \
  --dataset.root=/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real \
  --dataset.episode=0 \
  --dataset.fps=30
```

> **0.4.x 的欄位變化**(v0.6.1 → v0.4.3,branch `n1.7-graphen`):
> `bi_so_follower` → **`bi_so101_follower`**(本 repo 新增,見 [CALIBRATION.md](CALIBRATION.md) §1.2);
> `--robot.left_arm_config.port` → **`--robot.left_arm_port`**(扁平);
> **`use_degrees` 不用給**(預設 `False` = ±100,見 [CALIBRATION.md](CALIBRATION.md) §0)。
> `--dataset.*` 四個欄位(`repo_id` / `episode` / `root` / `fps`)在 0.4.3 完全相同。
>
> **`--dataset.fps` 要對上該資料集自己的 fps**(看 `meta/info.json`),不是照抄 30。

## 6. Hugging Face 下載 / 上傳

這版 lerobot 沒有專門的上傳/下載 CLI；下載靠 `hf download` (huggingface_hub CLI)，
上傳靠 Python `push_to_hub()`。
本機資料夾名 (底線) 與 HF 目標名 (連字號) 不同 — repo_id 只是上傳目標標籤，info.json 不存它，換名上傳沒問題。

```bash
# 一次性登入 (下載 private repo / 上傳都需要，寫入需 write token: https://huggingface.co/settings/tokens)
hf auth login
```

### 6a. 下載資料集 (download)

```bash
cd /home/graphen/sim2real/lerobot

# --repo-type dataset 必填 (預設是 model)；--local-dir 直接落地成平舖資料夾，
# 符合本專案 datasets/<name> 的慣例 (不要用預設 HF cache，list_episodes.py / delete_episodes.py 認的是這個路徑)
hf download ChihHanShen/bimanual-so101-pickvials-real-V2 \
  --repo-type dataset \
  --local-dir datasets/bimanual-so101-pickvials-real-V2

hf download ChihHanShen/bimanual-so101-pickvials-real-15fps \
  --repo-type dataset \
  --local-dir datasets/bimanual-so101-pickvials-real-15fps
```

> 資料夾若已存在，`hf download` 只會補缺少/變動的檔 (斷點續傳)，不會整包重下。

### 6b. 首次上傳 / 只是加更多集 (append)

```bash
cd /home/graphen/sim2real/lerobot
python -c '
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
python -c '
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
