#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True

    # By default `wrist_roll` is treated as a full-turn joint during calibration: its range is
    # hard-coded to the full encoder span (0..4095) and it is skipped in the range-of-motion sweep.
    # Set to `True` to instead sweep `wrist_roll` like every other joint and record its real min/max.
    # NOTE: with `use_degrees=True` (DEGREES norm mode) the recorded range only shifts the zero-point
    # midpoint and does not clip motion, so this mostly matters when `use_degrees=False`.
    calibrate_wrist_roll_range: bool = False


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig
