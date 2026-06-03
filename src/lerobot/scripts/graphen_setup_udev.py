#!/usr/bin/env python3
"""
Interactive udev rule setup tool for SO-101 arms.
Identifies each USB device by serial number and creates permanent symlinks.

Example:
    lerobot-setup-udev
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BY_ID_DIR = Path("/dev/serial/by-id")
UDEV_RULES_PATH = Path("/etc/udev/rules.d/99-lerobot.rules")
VENDOR_ID = "1a86"

DEVICE_ROLES = [
    ("follower_left", "Follower  左臂 (robot left arm)"),
    ("follower_right", "Follower  右臂 (robot right arm)"),
    ("leader_left", "Leader    左臂 (teleop left arm)"),
    ("leader_right", "Leader    右臂 (teleop right arm)"),
]


def get_by_id_devices() -> dict[str, str]:
    """Return {serial_number: ttyACMx}."""
    result = {}
    if not BY_ID_DIR.exists():
        return result
    for symlink in BY_ID_DIR.iterdir():
        name = symlink.name
        if "USB_Single_Serial_" not in name:
            continue
        serial = name.split("USB_Single_Serial_")[1].split("-")[0]
        target = symlink.resolve().name
        result[serial] = target
    return result


def wait_for_disconnect(before: dict[str, str]) -> str:
    """Wait for the user to unplug one device. Returns the disappeared serial number."""
    print("  請拔掉該設備的 USB，然後等待...")
    while True:
        time.sleep(0.5)
        after = get_by_id_devices()
        disappeared = set(before.keys()) - set(after.keys())
        if len(disappeared) == 1:
            return disappeared.pop()
        if len(disappeared) > 1:
            print(
                f"  [警告] 偵測到 {len(disappeared)} 個設備消失，請一次只拔一個。重新插回並再試。"
            )
            return None


def wait_for_reconnect(serial: str):
    """Wait for the user to replug the device."""
    print("  請重新插回 USB，然後等待...")
    while True:
        time.sleep(0.5)
        current = get_by_id_devices()
        if serial in current:
            print(f"  已偵測到設備重新連接 → /dev/{current[serial]}")
            return


def main():
    if os.geteuid() == 0:
        print("[警告] 不需要以 root 執行此腳本，產生規則時會自動使用 sudo。")

    print("=" * 60)
    print("  LeRobot udev 永久 Port 設定工具（雙臂）")
    print("=" * 60)
    print()

    before = get_by_id_devices()
    if not before:
        print("[錯誤] 找不到任何 /dev/serial/by-id 設備，請確認 USB 已連接。")
        sys.exit(1)

    print("目前已連接的設備：")
    for serial, tty in sorted(before.items(), key=lambda x: x[1]):
        print(f"  {serial}  →  /dev/{tty}")
    print()

    if len(before) < 4:
        print(
            f"[警告] 預期 4 個設備，目前只有 {len(before)} 個。請確認所有設備都已連接。"
        )
        input("確認後按 Enter 繼續，或 Ctrl+C 中止...")
        print()

    mapping: dict[str, str] = {}

    for role, description in DEVICE_ROLES:
        print(f"─── 設定：{description} ───")
        current = get_by_id_devices()

        serial = None
        while serial is None:
            serial = wait_for_disconnect(current)

        tty_before = current[serial]
        print(
            f"  識別到：序號 {serial}（原為 /dev/{tty_before}）→ 命名為 lerobot_{role}"
        )
        mapping[role] = serial

        wait_for_reconnect(serial)
        print()

    print("=" * 60)
    print("  產生 udev 規則")
    print("=" * 60)

    lines = [
        "# LeRobot 永久 USB Port 規則",
        "# 由 lerobot-setup-udev 自動產生",
        "",
    ]
    for role, serial in mapping.items():
        line = (
            f'SUBSYSTEM=="tty", '
            f'ATTRS{{idVendor}}=="{VENDOR_ID}", '
            f'ATTRS{{serial}}=="{serial}", '
            f'SYMLINK+="lerobot_{role}", MODE="0666"'
        )
        lines.append(line)
        print(f"  {line}")

    rules_content = "\n".join(lines) + "\n"

    print()
    print(f"寫入 {UDEV_RULES_PATH} ...")

    write_cmd = ["sudo", "tee", str(UDEV_RULES_PATH)]
    result = subprocess.run(
        write_cmd, input=rules_content.encode(), capture_output=True
    )
    if result.returncode != 0:
        print(f"[錯誤] 寫入失敗：{result.stderr.decode()}")
        sys.exit(1)

    print("套用規則...")
    subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=True)
    subprocess.run(["sudo", "udevadm", "trigger"], check=True)

    time.sleep(1)

    print()
    print("=" * 60)
    print("  完成！驗證結果：")
    print("=" * 60)
    for role, _ in DEVICE_ROLES:
        symlink = Path(f"/dev/lerobot_{role}")
        if symlink.exists():
            target = symlink.resolve().name
            print(f"  /dev/lerobot_{role}  →  /dev/{target}  ✓")
        else:
            print(f"  /dev/lerobot_{role}  →  [找不到，請重新插拔設備] ✗")

    print()
    print("之後命令改用固定名稱，例如：")
    print("  --robot.left_arm_config.port=/dev/lerobot_follower_left")
    print("  --robot.right_arm_config.port=/dev/lerobot_follower_right")
    print("  --teleop.left_arm_config.port=/dev/lerobot_leader_left")
    print("  --teleop.right_arm_config.port=/dev/lerobot_leader_right")
    print()
    print("udev 規則已設定 MODE=0666，插拔後不需要手動 chmod。")


if __name__ == "__main__":
    main()
