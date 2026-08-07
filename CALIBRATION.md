# SO-101 雙臂 Calibration 標準流程

這份文件的目的只有一個：**讓收資料時的 calibration 和 evaluation 時的 calibration 完全一致**，
並且在手臂維修過後，能夠客觀判斷「有沒有跑掉」、以及跑掉了要怎麼救。

相關檔案：

| 路徑 | 用途 |
| --- | --- |
| [calibration/bimanual_follower/](calibration/bimanual_follower/), [calibration/bimanual_leader/](calibration/bimanual_leader/) | 正在使用的 calibration（會被 `lerobot-calibrate` 覆寫） |
| [calibration/config/](calibration/config/) | `lerobot-calibrate` / `lerobot-record` 的 YAML，以及 `arm_serials.json`（USB 序號 → 手臂） |

**基準就是 git。** 這四個 JSON 已經在版本控制裡，不需要額外的備份機制 ——
「凍結」＝ commit，「檢查有沒有被改」＝ `git status`，「還原」＝ `git checkout`。

---

## 0. 為什麼 calibration 一變，訓練就對不上

Dataset 存的是**正規化後**的關節值。對同一個物理姿態，正規化的結果由
(`homing_offset`, `range_min`, `range_max`) 這組三元組決定
（[motors_bus.py:850-875](src/lerobot/motors/motors_bus.py#L850-L875)、
[feetech.py:278-288](src/lerobot/motors/feetech/feetech.py#L278-L288)）。

本 repo（v0.6.1）的 `use_degrees` 預設是 `True`，真機的 record/calibrate config 也明寫
`use_degrees: true`，所以五個身體關節走 **DEGREES** 模式，只有 gripper 走 RANGE_0_100：

```
raw = Actual_Position - Homing_Offset            # Feetech 韌體

# 五個身體關節（DEGREES）：尺度固定，只有零點會動
norm = (raw - (range_min + range_max)/2) * 360 / 4095

# gripper（RANGE_0_100）：尺度由 span 決定
norm = (clamp(raw) - range_min) / (range_max - range_min) * 100
```

兩個關鍵推論：

- **DEGREES 模式下 span 不影響尺度**（尺度固定是 360/4095 度 per count），
  真正決定對應關係的是**零點**：

  ```
  零點 Z = homing_offset + (range_min + range_max) / 2
  ```

  只要 `Z` 一樣，正規化就一樣。這也表示**只記 min/max 是不夠的** ——
  必須連 `homing_offset` 一起看。
- **`Z` 是錨在物理硬限位上的。** 掃描量到的 `range_min`/`range_max` 已經扣掉了
  `homing_offset`，所以只要兩次校準都掃到**同樣的物理硬限位**，`homing_offset`
  會完全抵消，`Z` 自動相同 —— 就算你每次「擺到行程中間」的位置都不一樣。

第二點很重要：它表示**一次認真的重新校準本來就會回到原本的正規化**，
不需要保存或搬移任何數字（見 §4）。校準之所以會失準，只有兩個原因：
**沒掃到真正的硬限位**，或**該關節根本沒有物理錨點**（§1.1 第 3 點的 `wrist_roll`）。

> ⚠️ **sim 端不是這個模式。** workshop 用的是 `~/lerobot-pinned`（v0.4.3），
> 那個版本的 `use_degrees` 預設是 **`False`** → **RANGE_M100_100**，而
> `utils/lerobot_interface.py` 也寫死了 `(raw+100)/200` 的假設。
> 所以**本文件的公式只適用於真機**；sim 資料集是另一套慣例。
> 兩者差的是純線性換算，GR00T 的 min-max 正規化會抵消掉，
> 所以各自 fine-tune 沒問題 —— 但 co-training 掛同一個 embodiment tag 就會出事
> （詳見 workshop `ROADMAP.md` 風險預警）。

另外，`lerobot-calibrate` 偵測到已有 calibration 檔時直接按 ENTER，
會把舊檔寫回馬達 EEPROM 而不重掃
（[so_follower.py:115-124](src/lerobot/robots/so_follower/so_follower.py#L115-L124)）。
沒事就走這條路，最安全。

---

## 1. 一次性：建立基準（只做一次，之後盡量不要再做）

**只有在準備開始收一批新資料時才做這一節。** 做完之後，這組 calibration 要陪
這批 dataset 走到 evaluation 結束。

### 1.1 校準時要注意的四件事

1. **「行程中間」要盡量準。** `homing_offset` 是 sign-magnitude 編碼、sign bit 在
   bit 11（[tables.py:209](src/lerobot/motors/feetech/tables.py#L209)），所以
   **|homing_offset| 上限是 2047**，超過會直接 `ValueError`。offset 是由
   `set_half_turn_homings()` 從你按 ENTER 當下的位置算出來的，離中間越遠值越大 ——
   擺得太偏會讓下次校準直接失敗。
2. **每個關節都要推到真正的機械硬限位。** 硬限位是物理定義的、可重現；
   「大概掃一掃」不可重現。整份檢查流程都建立在「`range_min`/`range_max` 就是硬限位」
   這個前提上 —— 掃不到底，後面所有比對都失去意義。
3. **`calibrate_wrist_roll_range: true` 四支都要開。** 預設是 `false`，會把
   `wrist_roll` 的範圍直接寫死成 0–4095 而不是掃出來的
   （[so_follower.py:151-152](src/lerobot/robots/so_follower/so_follower.py#L151-L152)）。
   四支只要有一支不一致，leader → follower 的該關節就會有 gain 誤差，
   而且那支關節在 §4 也失去可比對性。
4. **夾爪的「開到底 / 閉到底」四支要用同一套定義。** 否則左右夾爪的正規化尺度不同。

```bash
lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml
lerobot-calibrate --config_path=calibration/config/bimanual_so101_leader_config.yaml
```

### 1.2 立刻 commit

校準完馬上把四個 JSON 提交，這一個 commit 就是這批 dataset 的基準：

```bash
git add calibration/ && git commit -m "calib: baseline for <dataset name>"
```

**commit message 裡一定要寫上 dataset 名稱。** LeRobot 的 dataset 本身不會存
calibration，所以「這批資料是用哪組 calibration 收的」只存在於這行訊息裡 ——
之後要回溯就是靠 `git log calibration/`。

---

## 2. 每次開工前：兩個檢查

```bash
graphen-setup-udev              # USB：symlink 是否都在、序號是否正確
git status calibration/         # 檔案：應該是乾淨的，沒有任何 M
```

`git status` 只要出現 `M calibration/...`，就代表有人重跑過 `lerobot-calibrate`
把檔案蓋掉了。**不要接受它** —— 直接還原：

```bash
git diff calibration/           # 先看改了什麼
git checkout calibration/       # 還原成基準
```

### 馬達裡跑的是不是這份 calibration

lerobot 連線時會自動比對「檔案 vs 馬達 EEPROM」（`bus.is_calibrated`，比對
`Homing_Offset` / `Min_Position_Limit` / `Max_Position_Limit`）。兩邊一致就靜靜地過；
不一致才會跳出：

```
Press ENTER to use provided calibration file ... or type 'c' and press ENTER to run calibration:
```

所以規則很簡單：**正常開工時不該看到這個提示。看到了就代表有東西不對，先停下來查。**
確定檔案是對的之後，按 ENTER 會把檔案寫回馬達，這是正確的修復動作；
**千萬不要按 `c`**，那會重新校準並覆蓋基準。

> 這兩項都**驗不出機構位移**。機構有沒有跑掉要看 §4。

---

## 3. 什麼時候需要重新校準？

STS3215 用 12-bit 絕對式磁編碼器，raw count 綁在磁鐵相對輸出軸的物理位置。
**斷電、重開機、重插 USB 都不會掉。**

| 情況 | 要重新校準嗎 |
| --- | --- |
| 關機／重開機／重插 USB／換 USB 孔 | **不用** |
| 換 3D 列印外殼、連桿，但沒動到舵機與喇叭盤 | **不用**（不放心就走 §4 確認） |
| 把喇叭盤（horn）從輸出花鍵拆下來再裝回 | **要**（花鍵約 24 齒，差一齒 ≈ 15°） |
| 換一顆舵機 | **要** |
| 舵機 EEPROM factory reset／刷韌體／改 motor ID | **要** |
| 覺得「怪怪的」 | 先走 §4，用 `ΔZ` 找證據，不要憑感覺 |

重新校準本身**不是禁忌** —— 由 §0 可知，掃準硬限位的重校會回到同樣的正規化。
真正的風險是**掃不準**：沒推到底、每個關節用力程度不一，都會讓零點偏掉而且沒人會發現。
所以原則是：沒有具體理由就不要重跑；真的要跑，就照 §4 用 `ΔZ` 驗證結果。

---

## 4. 維修過後：直接重新校準，然後用 git diff 確認

不需要任何額外工具，也不需要把舊數字搬到新檔案裡。做法就是**正常重新校準**，
然後檢查零點有沒有變。

### 4.1 重新校準

```bash
lerobot-calibrate --config_path=calibration/config/bimanual_so101_follower_config.yaml
```

在提示時輸入 `c` 走完整流程。**掃描時務必推到真正的機械硬限位** ——
整個確認方法都建立在這一步上（§1.1 第 2 點）。

### 4.2 比較零點

```bash
git diff calibration/
```

對每個關節，用新舊兩組值各算一次零點：

```
Z = homing_offset + (range_min + range_max) / 2
```

然後看 `ΔZ = Z_新 − Z_舊`。因為尺度固定，**`ΔZ` 直接就是誤差**：

```
誤差角度 = ΔZ × 360 / 4095      （ΔZ = 30 counts ≈ 2.6°）
```

| 情況 | 意思 | 處理 |
| --- | --- | --- |
| `ΔZ` 在 ±30 counts 內 | 沒跑掉。校準前後等價 | `git checkout calibration/` 還原舊檔，保持基準乾淨 |
| `ΔZ` 明顯，但再掃一次數字不重複 | 掃描沒推到底 | **重掃**，不要採信第一次 |
| `ΔZ` 明顯，且重掃穩定重現 | 機構真的動了 | 新檔已經是對的 → §5 |
| 該關節 range 是 0–4095 | 沒有物理錨點（§1.1 第 3 點） | 比對無意義；先把 `calibrate_wrist_roll_range` 打開重校 |

gripper 要多看一項：它走 RANGE_0_100，**span 會直接改變尺度**。
所以 gripper 除了 `ΔZ`，`range_max − range_min` 也要和舊值接近。

> **為什麼不用把舊的 min/max 搬到新檔案？** 因為掃描量到的 range 已經扣掉了
> `homing_offset`，兩者一起看時 `homing_offset` 會抵消（§0）。反過來說，
> **把舊 range 貼到新檔案卻保留新的 `homing_offset` 是錯的** ——
> 那會讓零點偏掉整個 `homing_offset` 的差，實測可以差到上百度。
> 要嘛整份舊檔一起用（`git checkout`），要嘛整份新檔一起用，不要混。
>
> **搬「舊的 span」（`range_max − range_min` 的差值）也一樣沒用。**
> DEGREES 模式的公式裡根本沒有 span，只有中點；把新掃出來的 range 硬拉成舊 span，
> 若錨在中點則毫無效果，若錨在某一端反而會讓零點誤差**加倍**。
> span 不是可以移植的獨立量 —— 誤差在端點上，span 只是症狀。
> 唯一的例外：某一端因為機構問題已經掃不到了，而你確定**另一端**是準的，
> 這時可以錨在那個準的端點再套舊 span 把中點補回來。錨錯邊會更慘。

---

## 5. 真的跑掉了怎麼辦

`ΔZ` 穩定重現代表機構真的位移了。這時候要分清楚兩件事：

**新的 calibration 檔本身是正確的。** 硬限位的物理角度沒變，變的只是編碼器讀值；
重新掃描已經自動吸收掉這個位移，新檔對現在這支手臂是準的。

**但舊 dataset 不再對得上。** 那些資料是用舊的 `Z` 錄的，policy 學到的輸入分布
以舊 `Z` 為準。所以：

1. commit 新的 calibration，message 寫清楚是哪次維修造成的。
2. 評估 `ΔZ` 的大小（換算成角度）。幾度以內通常可以吞掉；
   一顆花鍵齒 ≈ 4096/24 ≈ **170 counts ≈ 15°**，那就不能當沒事。
3. 選一個：
   - 重收該手臂的資料（乾淨、推薦）；
   - 或接受 domain gap，並在 eval 報告裡註明 `ΔZ` 是多少。

**不要試圖把舊數字硬湊回新機構上。** 那會讓手臂靜靜地跑到錯誤的物理姿態 ——
新檔是準的，錯的是舊資料，要修的是資料不是校準。

---

## 6. USB 連線（udev）

四支手臂的控制板用的都是同一顆 USB-serial 晶片（`1a86:55d3` = WCH CH343），
`/dev/ttyACM*` 的編號會隨插拔與開機順序改變。
把 follower 的 calibration 寫進 leader 的馬達，就是從這裡開始出錯的。

規則檔在 `/etc/udev/rules.d/99-robot.rules`，建立固定名稱：

| symlink | 對應 |
| --- | --- |
| `/dev/ttyFollowerLeft` | follower 左臂 |
| `/dev/ttyFollowerRight` | follower 右臂 |
| `/dev/ttyLeaderLeft` | leader 左臂 |
| `/dev/ttyLeaderRight` | leader 右臂 |

這些名稱和 [calibration/config/](calibration/config/) 裡的 `port` 是一致的。

四支的 USB 序號存在
[calibration/config/arm_serials.json](calibration/config/arm_serials.json)，跟著 repo 走，
所以套用規則不需要任何互動：

```bash
graphen-setup-udev              # 檢查目前連線（預設，不會改動任何東西）
graphen-setup-udev --apply      # 依序號寫入 udev 規則（需要 sudo，會先備份舊規則）
graphen-setup-udev --identify   # 只有換過 USB 轉板才需要：重新辨識序號後自動套用
```

`--apply` 是冪等的：只要序號表沒變，重跑產生的規則就和現有的完全一樣。
換電腦或重灌時，`--apply` 一行就能把環境還原。

**規則必須綁 `ATTRS{serial}`（USB 序號），不能綁 `ATTRS{devpath}` 或 `ID_PATH`。**
綁序號才能做到「插哪個孔都對」；綁路徑的話換孔就會錯亂，而且會錯得很安靜。
規則也必須有 `MODE="0666"`，否則每次插拔都要手動 `chmod`。

`graphen-setup-udev` 的檢查除了比對序號，也會警告 `/etc/udev/rules.d/` 下同時存在
多份手臂規則檔 —— 兩份規則搶同一個 symlink 是很難查的問題。

### 序號屬於控制板，不屬於手臂

序號燒在控制板的 USB-serial 晶片裡。舵機只有 bus 內的 ID 1–6，四支手臂都一樣，
無法用來辨識。所以：

- 換外殼、換連桿、換舵機 → 序號不變，不用動 udev。
- **把控制板拆到另一支手臂上 → 身分跟著板子走**，固定名稱會指向錯的手臂，
  必須重跑 `graphen-setup-udev --identify`。

### port 和 calibration 檔是「約定」，不是自動對應

這點很容易誤會。calibration 檔的路徑由 config 的 `id` + 左右槽位決定
（[robot.py:53](src/lerobot/robots/robot.py#L53)：`calibration_dir / f"{id}.json"`），
**跟 port、跟 USB 序號完全無關**：

```
USB 序號 ──udev──▶ /dev/ttyFollowerLeft ──▶ YAML: left_arm_config.port
                                                 ⇕  只靠 YAML 寫得一致
YAML: id + "_left" ──▶ calibration_dir/<id>_left.json
```

兩段之間沒有任何機制在檢查。改錯 YAML 的 port、或 symlink 指錯，
**右手臂會載入左手臂的 calibration 而不會報錯**。

唯一的半套安全網是 `bus.is_calibrated`（檔案 vs 馬達 EEPROM 不符時 lerobot 會跳提示），
但**在那個提示按 ENTER 會把錯的 calibration 寫進馬達**，蓋掉原本正確的 —— 看起來最像
「預設繼續」的動作正是最糟的動作。所以要靠 §2 的兩個檢查把迴圈關起來，不要靠那個提示。

還有一種情況誰都偵測不到：把左右兩支手臂**連同控制板整組對調位置**。
電氣上全部正確、所有檢查都會過，但物理角色反了。只能靠實體標籤 + teleop 時目視確認。

---

## 7. NVIDIA 的 `so101_check_calibration.py` 是什麼定位

[Sim-to-Real-SO-101-Workshop/docker/real/scripts/so101_check_calibration.py](../Sim-to-Real-SO-101-Workshop/docker/real/scripts/so101_check_calibration.py)
是把你的 span 拿去比對 `calibration_stats.json` —— 那是**別人的一批手臂**的統計。

它回答的是「這次校準有沒有掃爛」，**不是**「這次校準跟我收資料時是不是同一個」。
它對 `homing_offset` 也只檢查絕對值上限，完全不管有沒有對齊。而且那個 baseline
的容差很緊，不同組裝批次很容易整批超標 —— 四支同時偏同一個方向時，
通常是 baseline 不合，不是四支都壞。

**當成「新手臂第一次校準完的 sanity check」用可以；跨時間的一致性檢查請用 §2 / §4。**
