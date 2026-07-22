#!/usr/bin/env python
"""Check that video-frame count == action/state-frame count for a LeRobotDataset.

Two layers:
  1. metadata consistency: per-episode `length` vs its video timestamp span * fps
  2. real files: ffprobe frame count of each .mp4 vs summed episode `length` for that file

Usage:
    python tools/check_frame_alignment.py <dataset_root>
    python tools/check_frame_alignment.py datasets/bimanual-so101-pickvials
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def ffprobe_nb_frames(mp4: Path) -> int:
    """Exact decoded frame count (slower but reliable across codecs)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else -1


def main(root: str) -> int:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    if not video_keys:
        print("No video features in this dataset — nothing to compare.")
        return 0

    ep_files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    print(f"dataset: {root.name}  fps={fps}  episodes={len(eps)}  video_keys={video_keys}\n")

    ok = True

    # ---- layer 1: per-episode metadata consistency (length vs timestamp span) ----
    print("== layer 1: per-episode  length  vs  round((to_ts - from_ts) * fps) ==")
    for vk in video_keys:
        c = f"videos/{vk}"
        bad = 0
        for _, e in eps.iterrows():
            span_frames = round((e[f"{c}/to_timestamp"] - e[f"{c}/from_timestamp"]) * fps)
            if span_frames != e["length"]:
                bad += 1
                print(f"  [MISMATCH] {vk} ep{int(e['episode_index'])}: "
                      f"length={int(e['length'])}  video_span_frames={span_frames}")
        print(f"  {vk}: {'OK' if bad == 0 else f'{bad} mismatched episodes'}")
        ok &= bad == 0

    # ---- layer 2: real .mp4 frame count vs summed episode length per file ----
    print("\n== layer 2: real ffprobe frame count  vs  sum(length) of episodes in that .mp4 ==")
    for vk in video_keys:
        ci, fi = f"videos/{vk}/chunk_index", f"videos/{vk}/file_index"
        missing, mism, checked = [], [], 0
        for (chunk, file_idx), grp in eps.groupby([ci, fi]):
            mp4 = root / "videos" / vk / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
            n_action = int(grp["length"].sum())
            if not mp4.exists():
                missing.append(mp4.name)
                continue
            checked += 1
            n_video = ffprobe_nb_frames(mp4)
            if n_video != n_action:
                mism.append(f"chunk-{int(chunk):03d}/file-{int(file_idx):03d}: "
                            f"video={n_video} action={n_action}")
        print(f"  {vk}: checked {checked} file(s), {len(mism)} mismatch, {len(missing)} missing-on-disk")
        for m in mism[:20]:
            print(f"     [MISMATCH] {m}")
        if missing:
            ok = False  # can't fully verify if files are absent
            print(f"     [MISSING] {len(missing)} referenced .mp4 not on disk "
                  f"(e.g. {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})")
        ok &= len(mism) == 0

    print("\n" + ("ALL ALIGNED ✅" if ok else
                  "MISALIGNMENT or INCOMPLETE — see above ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
