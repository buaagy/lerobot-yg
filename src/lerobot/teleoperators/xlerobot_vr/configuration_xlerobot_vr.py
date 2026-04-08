#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Optional

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_vr")
@dataclass
class XLerobotVRTeleopConfig(TeleoperatorConfig):
    # VR sysytem setting
    vr_enabled: bool = True
    vr_connection_timeout: float = 18000.0
    vr_data_timeout: float = 18000.0

    kp: float = 1.0
    enable_left_hand: bool = True
    enable_head: bool = True
    single_arm_mode: bool = False
    arm_controller: str = "right"

    arm_initial_x: float = 0.1629
    arm_initial_y: float = 0.1131
    arm_position_scale_x: float = 220.0
    arm_position_scale_y: float = 110.0
    arm_position_scale_z: float = 110.0
    arm_delta_position_scale: float = 0.02
    arm_delta_limit: float = 0.01
    arm_angle_scale: float = 2.0
    arm_angle_limit: float = 6.0
    arm_pan_scale: float = 180.0
    arm_ik_alpha: float = 0.27
    gripper_close_position: float = 45.0

    head_step_deg: float = 2.0
    head_zero_motor_1: float = 0.0
    head_zero_motor_2: float = 40.0

    base_kp: float = 1.0
    base_step: float = 0.1
    base_theta_step: float = 60.0
    base_thumb_deadzone: float = 0.1
    base_velocity_multiplier: float = 10.0

    event_thumb_threshold: float = 0.8
    event_thumb_reset_zone: float = 0.35
    event_thumb_axis_margin: float = 0.08

    xlevr_path: Optional[str] = "/your_local_DIR/XLeRobot/XLeVR"  # need to be modified
