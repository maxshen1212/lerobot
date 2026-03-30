#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    MultiDatasetNormalizerProcessorStep,
    MultiDatasetUnnormalizerProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    per_dataset_stats: list[dict[str, dict[str, torch.Tensor]]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    When ``per_dataset_stats`` is provided (multi-dataset co-training), the pipeline
    uses dataset-aware normalizers that apply per-sample normalization based on
    ``dataset_index``.  Otherwise the standard single-dataset normalizer is used.

    Args:
        config: The configuration object for the diffusion policy.
        dataset_stats: Single-dataset statistics for normalization.
        per_dataset_stats: Per-dataset statistics list for multi-dataset co-training.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    all_features = {**config.input_features, **config.output_features}

    if per_dataset_stats is not None:
        normalizer = MultiDatasetNormalizerProcessorStep(
            features=all_features,
            norm_map=config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
        )
        unnormalizer = MultiDatasetUnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
        )
    else:
        normalizer = NormalizerProcessorStep(
            features=all_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
        unnormalizer = UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        normalizer,
    ]
    output_steps = [
        unnormalizer,
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
