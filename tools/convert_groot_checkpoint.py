#!/usr/bin/env python
"""Convert an Isaac-GR00T (NVIDIA-native) N1.7 checkpoint into a LeRobot policy directory.

Why this exists: `lerobot-rollout` can only load a policy through `--policy.path`, which
goes through `PreTrainedConfig.from_pretrained` and therefore needs a LeRobot-style
`config.json` (with a `type` field) plus a `model.safetensors`. A raw Isaac-GR00T
checkpoint has neither, so it is rejected with `Missing 'type' field`. LeRobot *can*
read the raw checkpoint (`GrootPolicy` + the raw-N1.7 processor path), it just cannot be
pointed at it from the rollout CLI.

So this script builds the policy + processors from the raw checkpoint once, offline, and
saves them in LeRobot format:

    <out>/config.json                     # type=groot, base_model_path -> raw checkpoint
    <out>/model.safetensors               # single file (never sharded)
    <out>/policy_preprocessor.json/.safetensors
    <out>/policy_postprocessor.json/.safetensors

IMPORTANT: the saved config keeps `base_model_path` pointing at the raw checkpoint dir
(that is where the model architecture config lives), so *do not delete the raw
checkpoint* after converting.

Usage:
    uv run python tools/convert_groot_checkpoint.py \
        --raw-ckpt ~/models/bimanual-pickvials/pickvials-n1p7-run2/checkpoint-50000 \
        --dataset-root datasets/bimanual_so101_vial_pickplace_real \
        --out ~/models/bimanual-pickvials-lerobot/run2-50000
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import (
    GrootConfig,
    infer_groot_n1_7_action_horizon,
    is_raw_groot_n1_7_checkpoint,
)
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.policies.groot.utils import read_json
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-ckpt",
        required=True,
        help="Local Isaac-GR00T checkpoint dir (must contain config.json / processor_config.json / statistics.json)",
    )
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root (for features)")
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Dataset repo id; defaults to local/<dataset dir name> (only metadata is read from --dataset-root)",
    )
    parser.add_argument("--out", required=True, help="Output directory for the LeRobot policy")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Steps executed per inference; defaults to half the checkpoint action horizon, "
        "matching the official GR00T hardware recipe (chunk 16 / n_action_steps 8)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip the reload + round-trip inference check (not recommended)",
    )
    return parser.parse_args()


def checkpoint_modality(raw_ckpt: Path, embodiment_tag: str) -> dict:
    """Return the checkpoint's modality config for one embodiment, failing loudly if absent."""
    processor_config = read_json(raw_ckpt / "processor_config.json")
    processor_kwargs = processor_config.get("processor_kwargs", {})
    modality_configs = processor_kwargs.get("modality_configs", {})
    statistics = read_json(raw_ckpt / "statistics.json")

    missing = [
        name
        for name, container in (("processor_config.json", modality_configs), ("statistics.json", statistics))
        if embodiment_tag not in container
    ]
    if missing:
        raise SystemExit(
            f"embodiment tag '{embodiment_tag}' not found in {', '.join(missing)}.\n"
            f"  processor_config.json has: {sorted(modality_configs)}\n"
            f"  statistics.json has:       {sorted(statistics)}"
        )
    return {
        "processor_kwargs": processor_kwargs,
        "modality_config": modality_configs[embodiment_tag],
        "stats": statistics[embodiment_tag],
    }


def check_dataset_matches_checkpoint(ckpt: dict, ds_meta: LeRobotDatasetMetadata) -> None:
    """Fail fast when the dataset cannot feed this checkpoint (camera names / state layout)."""
    modality = ckpt["modality_config"]

    video_keys = list(modality.get("video", {}).get("modality_keys", []))
    dataset_cams = {k.removeprefix(f"{OBS_IMAGES}.") for k in ds_meta.features if k.startswith(OBS_IMAGES)}
    missing_cams = [k for k in video_keys if k not in dataset_cams]
    if missing_cams:
        raise SystemExit(
            f"checkpoint expects cameras {video_keys} but the dataset only has {sorted(dataset_cams)}.\n"
            f"missing: {missing_cams} (rollout would need --rename_map)"
        )

    state_keys = list(modality.get("state", {}).get("modality_keys", []))
    ckpt_state_dim = sum(len(ckpt["stats"]["state"][key]["min"]) for key in state_keys)
    dataset_state_dim = ds_meta.features[OBS_STATE]["shape"][0]
    if ckpt_state_dim != dataset_state_dim:
        raise SystemExit(
            f"state dim mismatch: checkpoint {state_keys} = {ckpt_state_dim}D, "
            f"dataset {OBS_STATE} = {dataset_state_dim}D"
        )

    print(f"  cameras     : {video_keys} (all present in dataset)")
    print(f"  state groups: {state_keys} = {ckpt_state_dim}D (matches dataset)")


def build_config(raw_ckpt: Path, ckpt: dict, ds_meta: LeRobotDatasetMetadata, args) -> GrootConfig:
    """Build a GrootConfig whose knobs are derived from the checkpoint, not hardcoded."""
    action_horizon = infer_groot_n1_7_action_horizon(raw_ckpt, args.embodiment_tag)
    if action_horizon is None:
        raise SystemExit(f"could not infer the action horizon for '{args.embodiment_tag}' from {raw_ckpt}")
    use_relative_actions = bool(ckpt["processor_kwargs"].get("use_relative_action", False))
    # https://huggingface.co/docs/lerobot/groot deploys SO-101 with chunk 16 / n_action_steps 8,
    # i.e. replan after half the predicted chunk (same cadence as NVIDIA's rollout wrapper).
    n_action_steps = args.n_action_steps or max(1, action_horizon // 2)

    cfg = GrootConfig(
        base_model_path=str(raw_ckpt),
        embodiment_tag=args.embodiment_tag,
        chunk_size=action_horizon,
        n_action_steps=n_action_steps,
        use_relative_actions=use_relative_actions,
        device=args.device,
    )

    # Same feature wiring as lerobot.policies.factory.make_policy.
    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    print(
        f"  action horizon      : {action_horizon} (chunk_size); executing {n_action_steps} steps/inference"
    )
    print(f"  use_relative_actions: {use_relative_actions} (from processor_config.json)")
    print(f"  action dim          : {cfg.output_features[ACTION].shape[0]}")
    return cfg


def infer_action_chunk(policy: GrootPolicy, preprocessor, postprocessor, batch: dict) -> torch.Tensor:
    """Run the chunked inference path (relative-action checkpoints cannot be stepped one frame at a time)."""
    torch.manual_seed(SEED)
    processed = preprocessor(dict(batch))
    chunk = policy.predict_action_chunk(processed)
    return postprocessor(chunk).detach().cpu()


def sample_batch(dataset: LeRobotDataset, device: str) -> dict:
    frame = dataset[0]
    batch = {k: v for k, v in frame.items() if isinstance(v, torch.Tensor)}
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["task"] = frame["task"]
    return batch


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    raw_ckpt = Path(args.raw_ckpt).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    repo_id = args.dataset_repo_id or f"local/{dataset_root.name}"

    print(f"raw checkpoint : {raw_ckpt}")
    print(f"dataset        : {dataset_root}")
    print(f"output         : {out_dir}")

    if not is_raw_groot_n1_7_checkpoint(raw_ckpt):
        raise SystemExit(
            f"{raw_ckpt} is not a raw GR00T N1.7 checkpoint (expected a config.json with no 'type' field "
            "and architectures = ['Gr00tN1d7']). Nothing to convert."
        )

    print("\n[1/5] checking checkpoint <-> dataset compatibility")
    ckpt = checkpoint_modality(raw_ckpt, args.embodiment_tag)
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    check_dataset_matches_checkpoint(ckpt, ds_meta)

    print("\n[2/5] building policy config")
    cfg = build_config(raw_ckpt, ckpt, ds_meta, args)

    print("\n[3/5] loading GR00T weights from the raw checkpoint")
    policy = GrootPolicy(cfg)
    policy.eval()
    policy.to(cfg.device)

    # No dataset_stats / dataset_meta: that would flip the pipeline into training mode
    # (state dropout, image augmentation) and normalize with dataset stats instead of the
    # checkpoint's baked-in ones.
    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=cfg, dataset_stats=None, dataset_meta=None
    )

    print(f"\n[4/5] saving LeRobot policy to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    preprocessor.save_pretrained(out_dir)
    postprocessor.save_pretrained(out_dir)
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    if args.no_smoke_test:
        print("\n[5/5] smoke test skipped (--no-smoke-test)")
        return 0

    print("\n[5/5] smoke test: round-trip through the rollout load path")
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[0])
    batch = sample_batch(dataset, cfg.device)
    before = infer_action_chunk(policy, preprocessor, postprocessor, batch)
    print(f"  pre-save  chunk: shape={tuple(before.shape)} range=[{before.min():.3f}, {before.max():.3f}]")

    del policy, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()

    # Exactly what lerobot-rollout does: RolloutConfig.__post_init__ -> context.build_rollout_context.
    reloaded_cfg = PreTrainedConfig.from_pretrained(out_dir)
    reloaded_cfg.pretrained_path = str(out_dir)
    reloaded_policy = GrootPolicy.from_pretrained(out_dir, config=reloaded_cfg)
    reloaded_policy.to(reloaded_cfg.device)
    reloaded_pre, reloaded_post = make_pre_post_processors(
        policy_cfg=reloaded_cfg,
        pretrained_path=str(out_dir),
        preprocessor_overrides={"device_processor": {"device": reloaded_cfg.device}},
    )
    after = infer_action_chunk(reloaded_policy, reloaded_pre, reloaded_post, batch)
    print(f"  reloaded  chunk: shape={tuple(after.shape)} range=[{after.min():.3f}, {after.max():.3f}]")

    action_dim = reloaded_cfg.output_features[ACTION].shape[0]
    expected = (1, reloaded_cfg.n_action_steps, action_dim)
    if tuple(after.shape) != expected:
        raise SystemExit(f"unexpected action chunk shape {tuple(after.shape)}, expected {expected}")
    if torch.isnan(after).any():
        raise SystemExit("reloaded policy produced NaNs")
    max_diff = (before - after).abs().max().item()
    print(f"  max |pre-save - reloaded| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit("round-trip mismatch: the saved checkpoint does not reproduce the source model")

    print(f"\nOK. Use it with: lerobot-rollout --policy.path={out_dir} --inference.type=rtc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
