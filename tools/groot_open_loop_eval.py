#!/usr/bin/env python
"""Open-loop check of a converted GR00T policy against a recorded LeRobot dataset.

Run this before putting a sim-trained GR00T checkpoint on the real robot. It answers two
questions offline:

1. **Is the real robot even inside the training distribution?**  The checkpoint normalizes
   state/action with the statistics baked into `statistics.json` (percentiles q01/q99, with
   `clip_outliers`). If the real joint values fall outside the sim range they get clipped and
   the policy sees a saturated observation. `--stats-only` reports that per joint without
   loading the model.
2. **Does the policy predict the demonstrated actions?**  For sampled frames it runs the same
   preprocess -> `predict_action_chunk` -> postprocess path that `lerobot-rollout --inference.type=rtc`
   uses, and compares the decoded chunk against the recorded actions. A "hold current state"
   baseline is printed alongside so the MAE numbers have a scale.

Inference goes through the chunk API on purpose: relative-action checkpoints cannot be decoded
one step at a time (`GrootPolicy.select_action` raises for them).

Usage:
    uv run python tools/groot_open_loop_eval.py \
        --policy-dir ~/models/bimanual-pickvials-lerobot/run2-50000 \
        --dataset-root datasets/bimanual_so101_vial_pickplace_real \
        --episodes 0,1,2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.utils import read_json
from lerobot.utils.constants import ACTION, OBS_STATE

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-dir", required=True, help="Converted LeRobot policy directory")
    parser.add_argument(
        "--dataset-root", default=None, help="Local dataset root; omit to pull --dataset-repo-id from the Hub"
    )
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--episodes", default="0", help="Comma-separated episode indices")
    parser.add_argument("--stride", type=int, default=20, help="Sample every Nth frame")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stats-only", action="store_true", help="Only run the distribution check")
    return parser.parse_args()


def load_policy_config(policy_dir: Path) -> PreTrainedConfig:
    cfg = PreTrainedConfig.from_pretrained(policy_dir)
    cfg.pretrained_path = str(policy_dir)
    return cfg


def state_group_layout(raw_ckpt: Path, embodiment_tag: str) -> tuple[list[str], dict]:
    """Return the checkpoint's state modality keys and its per-group statistics."""
    processor_config = read_json(raw_ckpt / "processor_config.json")
    modality = processor_config["processor_kwargs"]["modality_configs"][embodiment_tag]
    stats = read_json(raw_ckpt / "statistics.json")[embodiment_tag]["state"]
    return list(modality["state"]["modality_keys"]), stats


def flat_state_bounds(
    state_keys: list[str], stats: dict, use_percentiles: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the checkpoint's grouped state bounds into one vector, in checkpoint order."""
    low_name, high_name = ("q01", "q99") if use_percentiles else ("min", "max")
    low = np.concatenate([np.asarray(stats[key][low_name], dtype=np.float64) for key in state_keys])
    high = np.concatenate([np.asarray(stats[key][high_name], dtype=np.float64) for key in state_keys])
    return low, high


def report_distribution(dataset: LeRobotDataset, cfg, joint_names: list[str]) -> None:
    """Report how much of the recorded state falls outside the checkpoint's normalization range."""
    raw_ckpt = Path(cfg.base_model_path).expanduser()
    state_keys, stats = state_group_layout(raw_ckpt, cfg.embodiment_tag)
    processor_kwargs = read_json(raw_ckpt / "processor_config.json")["processor_kwargs"]
    use_percentiles = bool(processor_kwargs.get("use_percentiles", False))
    clip_outliers = bool(processor_kwargs.get("clip_outliers", False))
    low, high = flat_state_bounds(state_keys, stats, use_percentiles)

    state = np.stack(dataset.hf_dataset.with_format("numpy")[OBS_STATE])
    below = (state < low).mean(axis=0) * 100
    above = (state > high).mean(axis=0) * 100

    bound_label = "q01/q99" if use_percentiles else "min/max"
    print(f"\n=== state distribution vs checkpoint {bound_label} (clip_outliers={clip_outliers}) ===")
    print(f"{'joint':<24}{'ckpt low':>10}{'ckpt high':>11}{'real min':>10}{'real max':>10}{'% clipped':>11}")
    for i, name in enumerate(joint_names):
        clipped = below[i] + above[i]
        flag = "  <-- " if clipped > 1.0 else ""
        print(
            f"{name:<24}{low[i]:>10.2f}{high[i]:>11.2f}{state[:, i].min():>10.2f}"
            f"{state[:, i].max():>10.2f}{clipped:>10.1f}%{flag}"
        )
    total = ((state < low) | (state > high)).mean() * 100
    print(f"{'ALL':<24}{'':>10}{'':>11}{'':>10}{'':>10}{total:>10.1f}%")


def evaluate_open_loop(dataset: LeRobotDataset, policy, preprocessor, postprocessor, args, joint_names):
    indices = list(range(0, len(dataset), args.stride))[: args.max_samples]
    horizon = None

    errors, baseline_errors, first_step, last_step, skipped = [], [], [], [], 0
    for count, idx in enumerate(indices, start=1):
        frame = dataset[idx]
        if bool(frame[f"{ACTION}_is_pad"].any()):
            skipped += 1
            continue

        batch = {k: v.to(args.device) for k, v in frame.items() if isinstance(v, torch.Tensor)}
        batch["task"] = frame["task"]

        torch.manual_seed(SEED)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(preprocessor(dict(batch)))
            predicted = postprocessor(chunk).detach().cpu().squeeze(0)

        # The policy only returns n_action_steps of the chunk (official recipe replans at half
        # the horizon), so compare over whatever it actually predicts.
        horizon = predicted.shape[0]
        ground_truth = frame[ACTION].cpu()
        error = (predicted[:horizon] - ground_truth[:horizon]).abs().numpy()
        errors.append(error)
        first_step.append(error[0])
        last_step.append(error[-1])
        # Baseline: do nothing, i.e. command the current joint positions for the whole chunk.
        hold = frame[OBS_STATE].cpu().unsqueeze(0).expand_as(ground_truth[:horizon])
        baseline_errors.append((hold - ground_truth[:horizon]).abs().numpy())
        print(f"  [{count}/{len(indices)}] frame {idx}: MAE {error.mean():.3f}", end="\r", flush=True)

    if not errors:
        raise SystemExit("no usable samples (every sampled frame had a padded action chunk)")

    errors = np.stack(errors)
    baseline_errors = np.stack(baseline_errors)
    print(
        f"\n\n=== open-loop action error over {len(errors)} chunks of {horizon} steps "
        f"({skipped} skipped for padding) ==="
    )
    print(f"{'joint':<24}{'MAE':>9}{'step 1':>9}{'step ' + str(horizon):>9}{'hold-state MAE':>17}")
    per_joint = errors.mean(axis=(0, 1))
    first = np.stack(first_step).mean(axis=0)
    last = np.stack(last_step).mean(axis=0)
    baseline_per_joint = baseline_errors.mean(axis=(0, 1))
    for i, name in enumerate(joint_names):
        print(f"{name:<24}{per_joint[i]:>9.3f}{first[i]:>9.3f}{last[i]:>9.3f}{baseline_per_joint[i]:>17.3f}")
    print(
        f"{'ALL':<24}{errors.mean():>9.3f}{first.mean():>9.3f}{last.mean():>9.3f}"
        f"{baseline_errors.mean():>17.3f}"
    )
    print("\n(units are LeRobot normalized joint positions; 'hold-state' = command the current pose,")
    print(" so the policy is only informative if its MAE is clearly below that column.)")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    policy_dir = Path(args.policy_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    repo_id = args.dataset_repo_id or (f"local/{dataset_root.name}" if dataset_root else None)
    if repo_id is None:
        raise SystemExit("pass --dataset-root and/or --dataset-repo-id")
    episodes = [int(e) for e in args.episodes.split(",") if e.strip()]

    cfg = load_policy_config(policy_dir)
    if cfg.type != "groot":
        raise SystemExit(f"{policy_dir} holds a '{cfg.type}' policy, not groot")
    print(f"policy    : {policy_dir}")
    print(f"base model: {cfg.base_model_path}")
    print(f"embodiment: {cfg.embodiment_tag} | chunk {cfg.chunk_size} | relative={cfg.use_relative_actions}")

    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=episodes)
    joint_names = dataset.meta.features[OBS_STATE]["names"]
    report_distribution(dataset, cfg, joint_names)
    if args.stats_only:
        return 0

    fps = dataset.meta.fps
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps={ACTION: [k / fps for k in range(cfg.chunk_size)]},
    )

    print(f"\nloading policy on {args.device} ...")
    cfg.device = args.device
    policy = GrootPolicy.from_pretrained(policy_dir, config=cfg)
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    evaluate_open_loop(dataset, policy, preprocessor, postprocessor, args, joint_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
