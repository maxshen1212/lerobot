#!/usr/bin/env python
"""Inspect a single episode of a LeRobotDataset by episode_index.

A LeRobot v3 dataset packs MANY episodes into one .mp4 (named by *file number*,
not episode_index). This tool maps an episode_index -> (mp4 file, time span) for
each camera and, by default, extracts that span with ffmpeg and opens it.

Usage:
    # list every episode -> file + time span (all cameras' center reference)
    python tools/view_episode.py <dataset_root> --list

    # extract + open one episode across all cameras
    python tools/view_episode.py <dataset_root> 47

    # one camera only, don't auto-open
    python tools/view_episode.py <dataset_root> 47 --cam center --no-open

Examples:
    python tools/view_episode.py datasets/bimanual_so101_vial_pickplace_real --list
    python tools/view_episode.py datasets/bimanual_so101_vial_pickplace_real 47
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    ep_files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True).set_index("episode_index")
    return info, fps, video_keys, eps


def locate(eps, vk: str, ep: int):
    """Return (chunk_index, file_index, from_s, to_s) for episode `ep` and video key `vk`."""
    row = eps.loc[ep]
    c = f"videos/{vk}"
    return (int(row[f"{c}/chunk_index"]), int(row[f"{c}/file_index"]),
            float(row[f"{c}/from_timestamp"]), float(row[f"{c}/to_timestamp"]))


def mp4_path(root: Path, vk: str, chunk: int, file_idx: int) -> Path:
    return root / "videos" / vk / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"


def do_list(root: Path, eps, video_keys, fps):
    vk = video_keys[0]
    print(f"episode -> mp4 file / time span   (reference camera: {vk})\n")
    print("%4s %7s  %-18s %8s %8s" % ("ep", "len", "mp4 file", "from_s", "to_s"))
    for ep in sorted(eps.index):
        ch, fi, a, b = locate(eps, vk, ep)
        flag = "  <- 3-frame stub!" if int(eps.loc[ep]["length"]) < fps else ""
        print("%4d %7d  chunk-%03d/file-%03d %8.1f %8.1f%s"
              % (ep, int(eps.loc[ep]["length"]), ch, fi, a, b, flag))


def do_view(root: Path, eps, video_keys, ep: int, only_cam: str | None, do_open: bool):
    keys = video_keys if only_cam is None else [k for k in video_keys if k.endswith(only_cam)]
    if not keys:
        sys.exit(f"camera '{only_cam}' not found among {video_keys}")
    out_dir = Path("/tmp") / f"ep{ep}_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    for vk in keys:
        ch, fi, a, b = locate(eps, vk, ep)
        src = mp4_path(root, vk, ch, fi)
        cam = vk.split(".")[-1]
        dst = out_dir / f"ep{ep}_{cam}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ss", f"{a}", "-to", f"{b}", "-c", "copy", str(dst)], check=True)
        print(f"  {cam:14s} {src.name} [{a:.1f}s-{b:.1f}s] -> {dst}")
        if do_open:
            subprocess.Popen(["xdg-open", str(dst)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\nclips in {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="dataset root (folder with meta/ data/ videos/)")
    p.add_argument("episode", nargs="?", type=int, help="episode_index to extract")
    p.add_argument("--list", action="store_true", help="list all episodes -> file/time, then exit")
    p.add_argument("--cam", help="only this camera (e.g. center, wrist_left, wrist_right)")
    p.add_argument("--no-open", action="store_true", help="extract clips but don't open them")
    args = p.parse_args()

    info, fps, video_keys, eps = load(args.root)
    if not video_keys:
        sys.exit("dataset has no video features")

    if args.list or args.episode is None:
        do_list(args.root, eps, video_keys, fps)
        return 0
    if args.episode not in eps.index:
        sys.exit(f"episode {args.episode} not in dataset (have {eps.index.min()}..{eps.index.max()})")
    do_view(args.root, eps, video_keys, args.episode, args.cam, not args.no_open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
