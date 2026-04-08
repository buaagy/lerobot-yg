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

import asyncio
import logging
import threading
import time
from typing import Any

from lerobot.model.SO101Robot import SO101Kinematics

from ..teleoperator import Teleoperator
from .configuration_xlerobot_vr import XLerobotVRTeleopConfig

logger = logging.getLogger(__name__)

VR_AVAILABLE = True
try:
    from .vr_monitor import VRMonitor
except Exception as exc:  # pragma: no cover - depends on local VR runtime
    VR_AVAILABLE = False
    VRMonitor = None
    logger.warning(f"VR Monitor not available: {exc}")


LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}

RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

STATE_FEATURES = {
    "left_arm_shoulder_pan.pos": float,
    "left_arm_shoulder_lift.pos": float,
    "left_arm_elbow_flex.pos": float,
    "left_arm_wrist_flex.pos": float,
    "left_arm_wrist_roll.pos": float,
    "left_arm_gripper.pos": float,
    "right_arm_shoulder_pan.pos": float,
    "right_arm_shoulder_lift.pos": float,
    "right_arm_elbow_flex.pos": float,
    "right_arm_wrist_flex.pos": float,
    "right_arm_wrist_roll.pos": float,
    "right_arm_gripper.pos": float,
    "head_motor_1.pos": float,
    "head_motor_2.pos": float,
    "x.vel": float,
    "y.vel": float,
    "theta.vel": float,
}

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def resolve_arm_joint_map(robot_obs: dict[str, Any], side: str | None = None) -> dict[str, str]:
    joint_map = {}
    for joint_name in ARM_JOINT_NAMES:
        candidates = []
        if side:
            candidates.extend(
                (
                    f"{side}_arm_{joint_name}.pos",
                    f"{side}_{joint_name}.pos",
                )
            )
        candidates.append(f"{joint_name}.pos")
        for key in candidates:
            if key in robot_obs:
                joint_map[joint_name] = key.removesuffix(".pos")
                break

    missing = [joint_name for joint_name in ARM_JOINT_NAMES if joint_name not in joint_map]
    if missing:
        arm_label = side if side else "single"
        raise RuntimeError(
            f"Could not resolve {arm_label} arm joint keys from robot observation. Missing: {missing}"
        )

    return joint_map


class SimpleTeleopArm:
    def __init__(
        self,
        joint_map: dict[str, str],
        initial_obs: dict[str, Any],
        kinematics,
        prefix: str,
        config: XLerobotVRTeleopConfig,
    ):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = config.kp
        self.kinematics = kinematics
        self.config = config

        self.current_x = config.arm_initial_x
        self.current_y = config.arm_initial_y
        self.pitch = 0.0
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = self.target_positions.copy()

        for joint_name, motor_name in joint_map.items():
            self.target_positions[joint_name] = float(initial_obs.get(f"{motor_name}.pos", 0.0))

        self.prev_vr_pos = None
        self.prev_wrist_flex = None
        self.prev_wrist_roll = None

    def move_to_zero_position(self) -> dict[str, float]:
        self.target_positions = self.zero_pos.copy()
        self.current_x = self.config.arm_initial_x
        self.current_y = self.config.arm_initial_y
        self.pitch = 0.0
        self.prev_vr_pos = None
        self.prev_wrist_flex = None
        self.prev_wrist_roll = None
        self.target_positions["wrist_flex"] = 0.0
        return {
            f"{motor_name}.pos": self.target_positions[joint_name]
            for joint_name, motor_name in self.joint_map.items()
        }

    def handle_vr_input(self, vr_goal) -> None:
        if vr_goal is None or getattr(vr_goal, "target_position", None) is None:
            return

        current_vr_pos = vr_goal.target_position
        if self.prev_vr_pos is None:
            self.prev_vr_pos = current_vr_pos
            return

        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * self.config.arm_position_scale_x
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * self.config.arm_position_scale_y
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * self.config.arm_position_scale_z
        self.prev_vr_pos = current_vr_pos

        pos_scale = self.config.arm_delta_position_scale
        angle_scale = self.config.arm_angle_scale
        delta_limit = self.config.arm_delta_limit
        angle_limit = self.config.arm_angle_limit

        delta_x = max(-delta_limit, min(delta_limit, vr_x * pos_scale))
        delta_y = max(-delta_limit, min(delta_limit, vr_y * pos_scale))
        delta_z = max(-delta_limit, min(delta_limit, vr_z * pos_scale))

        self.current_x += -delta_z
        self.current_y += delta_y

        wrist_flex_deg = getattr(vr_goal, "wrist_flex_deg", None)
        if wrist_flex_deg is not None:
            if self.prev_wrist_flex is None:
                self.prev_wrist_flex = wrist_flex_deg
            else:
                delta_pitch = (wrist_flex_deg - self.prev_wrist_flex) * angle_scale
                delta_pitch = max(-angle_limit, min(angle_limit, delta_pitch))
                self.pitch = max(-90, min(90, self.pitch + delta_pitch))
                self.prev_wrist_flex = wrist_flex_deg

        wrist_roll_deg = getattr(vr_goal, "wrist_roll_deg", None)
        if wrist_roll_deg is not None:
            if self.prev_wrist_roll is None:
                self.prev_wrist_roll = wrist_roll_deg
            else:
                delta_roll = (wrist_roll_deg - self.prev_wrist_roll) * angle_scale
                delta_roll = max(-angle_limit, min(angle_limit, delta_roll))
                current_roll = self.target_positions.get("wrist_roll", 0.0)
                self.target_positions["wrist_roll"] = max(-90, min(90, current_roll + delta_roll))
                self.prev_wrist_roll = wrist_roll_deg

        if abs(delta_x) > 0.001:
            delta_pan = max(-angle_limit, min(angle_limit, delta_x * self.config.arm_pan_scale))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            self.target_positions["shoulder_pan"] = max(-180, min(180, current_pan + delta_pan))

        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            alpha = self.config.arm_ik_alpha
            self.target_positions["shoulder_lift"] = (1 - alpha) * self.target_positions["shoulder_lift"] + alpha * joint2_target
            self.target_positions["elbow_flex"] = (1 - alpha) * self.target_positions["elbow_flex"] + alpha * joint3_target
        except Exception as exc:
            logger.debug(f"[{self.prefix}] VR IK failed: {exc}")

        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch
        )
        self.target_positions["gripper"] = (
            self.config.gripper_close_position if vr_goal.metadata.get("trigger", 0) > 0.5 else 0.0
        )

    def p_control_action(self, robot_obs: dict[str, Any]) -> dict[str, float]:
        action = {}
        for joint_name, motor_name in self.joint_map.items():
            current = float(robot_obs.get(f"{motor_name}.pos", 0.0))
            error = self.target_positions[joint_name] - current
            action[f"{motor_name}.pos"] = current + self.kp * error
        return action


class SimpleHeadControl:
    def __init__(self, initial_obs: dict[str, Any], config: XLerobotVRTeleopConfig):
        self.kp = config.kp
        self.degree_step = config.head_step_deg
        self.config = config
        self.target_positions = {
            "head_motor_1": float(initial_obs.get("head_motor_1.pos", 0.0)),
            "head_motor_2": float(initial_obs.get("head_motor_2.pos", config.head_zero_motor_2)),
        }
        self.zero_pos = {
            "head_motor_1": config.head_zero_motor_1,
            "head_motor_2": config.head_zero_motor_2,
        }

    def handle_vr_input(self, vr_goal) -> None:
        if vr_goal is None:
            return
        thumb = vr_goal.metadata.get("thumbstick", {})
        thumb_x = thumb.get("x", 0.0)
        thumb_y = thumb.get("y", 0.0)
        if abs(thumb_x) > 0.1:
            self.target_positions["head_motor_1"] += self.degree_step if thumb_x > 0 else -self.degree_step
        if abs(thumb_y) > 0.1:
            self.target_positions["head_motor_2"] += self.degree_step if thumb_y > 0 else -self.degree_step

    def move_to_zero_position(self) -> dict[str, float]:
        self.target_positions = self.zero_pos.copy()
        return {f"{motor}.pos": value for motor, value in self.target_positions.items()}

    def p_control_action(self, robot_obs: dict[str, Any]) -> dict[str, float]:
        action = {}
        for motor, target in self.target_positions.items():
            current = float(robot_obs.get(f"{motor}.pos", 0.0))
            action[f"{motor}.pos"] = current + self.kp * (target - current)
        return action


class SimpleBaseControl:
    def __init__(self, initial_obs: dict[str, Any], config: XLerobotVRTeleopConfig):
        self.kp = config.base_kp
        self.config = config
        self.target_velocities = {
            "x.vel": float(initial_obs.get("x.vel", 0.0)),
            "y.vel": float(initial_obs.get("y.vel", 0.0)),
            "theta.vel": float(initial_obs.get("theta.vel", 0.0)),
        }
        self.zero_vel = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    def handle_vr_input(self, vr_goal) -> None:
        thumb = vr_goal.metadata.get("thumbstick", {}) if vr_goal else {}
        buttons = vr_goal.metadata.get("buttons", {}) if vr_goal else {}
        twist = vr_goal.metadata.get("twist", 0.0) if vr_goal else 0.0

        thumb_x = thumb.get("x", 0.0)
        thumb_y = thumb.get("y", 0.0)
        deadzone = self.config.base_thumb_deadzone
        step = self.config.base_step
        theta_step = self.config.base_theta_step
        multiplier = self.config.base_velocity_multiplier

        self.target_velocities["y.vel"] = -thumb_x * step * multiplier if abs(thumb_x) > deadzone else 0.0
        self.target_velocities["x.vel"] = -thumb_y * step * multiplier if abs(thumb_y) > deadzone else 0.0
        self.target_velocities["theta.vel"] = twist * theta_step * multiplier if abs(twist) > deadzone else 0.0

        if buttons.get("a", False):
            self.target_velocities["theta.vel"] = -theta_step
        if buttons.get("b", False):
            self.target_velocities["theta.vel"] = theta_step

    def move_to_zero_velocity(self) -> dict[str, float]:
        self.target_velocities = self.zero_vel.copy()
        return self.zero_vel.copy()

    def p_control_action(self, robot_obs: dict[str, Any]) -> dict[str, float]:
        action = {}
        for key, target in self.target_velocities.items():
            current = float(robot_obs.get(key, 0.0))
            action[key] = current + self.kp * (target - current)
        return action


class VREventHandler:
    def __init__(self, config: XLerobotVRTeleopConfig):
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "reset_position": False,
        }
        self.threshold = config.event_thumb_threshold
        self.reset_zone = config.event_thumb_reset_zone
        self.axis_margin = config.event_thumb_axis_margin
        self.armed = True

    def process_left_controller(self, metadata: dict[str, Any]) -> None:
        thumb = metadata.get("thumbstick", {})
        thumb_x = thumb.get("x", 0.0)
        thumb_y = thumb.get("y", 0.0)
        abs_x = abs(thumb_x)
        abs_y = abs(thumb_y)

        if abs_x < self.reset_zone and abs_y < self.reset_zone:
            self.armed = True
            self.events["reset_position"] = False
            return

        if not self.armed:
            self.events["reset_position"] = False
            return

        # Trigger only on clear cardinal directions.
        # If both axes are too close, treat as diagonal/ambiguous and ignore.
        if abs_x >= self.threshold or abs_y >= self.threshold:
            if abs(abs_x - abs_y) < self.axis_margin:
                self.events["reset_position"] = False
                return

        if abs_x > abs_y and abs_x >= self.threshold:
            if thumb_x > 0:
                # End current episode early and keep recording next episodes.
                self.events["exit_early"] = True
                logger.info("[VR Event] End current episode early requested.")
            else:
                # Re-record current episode.
                self.events["rerecord_episode"] = True
                self.events["exit_early"] = True
                logger.info("[VR Event] Re-record current episode requested.")
            self.armed = False
            self.events["reset_position"] = False
            return

        if abs_y > abs_x and abs_y >= self.threshold:
            if thumb_y < 0:
                self.events["reset_position"] = True
                logger.info("[VR Event] Reset robot position requested.")
            else:
                # Stop recording and discard current in-progress episode.
                self.events["stop_recording"] = True
                self.events["rerecord_episode"] = True
                self.events["exit_early"] = True
                logger.info("[VR Event] Stop recording and discard current episode requested.")
            self.armed = False
            return

        self.events["reset_position"] = False

    def get_events(self) -> dict[str, bool]:
        return self.events.copy()

    def reset_latched_events(self) -> None:
        self.events["exit_early"] = False
        self.events["rerecord_episode"] = False
        self.events["stop_recording"] = False
        self.events["reset_position"] = False


class XLerobotVRTeleop(Teleoperator):
    config_class = XLerobotVRTeleopConfig
    name = "xlerobot_vr"

    def __init__(self, config: XLerobotVRTeleopConfig):
        super().__init__(config)
        self.config = config
        self.vr_monitor = None
        self.vr_thread = None
        self.logs = {}
        self._connected = False
        self._calibrated = False
        self.latest_robot_obs: dict[str, Any] | None = None
        self.primary_arm: SimpleTeleopArm | None = None
        self.left_arm: SimpleTeleopArm | None = None
        self.right_arm: SimpleTeleopArm | None = None
        self.head_control: SimpleHeadControl | None = None
        self.base_control: SimpleBaseControl | None = None
        self.vr_event_handler = VREventHandler(config)

    def _resolve_controlled_arm(self) -> tuple[dict[str, str], str]:
        if self.latest_robot_obs is None:
            raise RuntimeError("No robot observation available yet.")

        preferred_side = (self.config.arm_controller or "right").lower()
        if preferred_side not in {"left", "right"}:
            raise ValueError(
                f"Unsupported arm_controller={self.config.arm_controller!r}. Expected 'left' or 'right'."
            )

        errors: list[str] = []
        for side in (preferred_side, "left" if preferred_side == "right" else "right"):
            try:
                return resolve_arm_joint_map(self.latest_robot_obs, side), side
            except RuntimeError as exc:
                errors.append(str(exc))

        try:
            return resolve_arm_joint_map(self.latest_robot_obs, None), preferred_side
        except RuntimeError as exc:
            errors.append(str(exc))

        raise RuntimeError(" ; ".join(errors))

    @property
    def action_features(self) -> dict[str, type]:
        return STATE_FEATURES.copy()

    @property
    def feedback_features(self) -> dict[str, type]:
        return STATE_FEATURES.copy()

    @property
    def is_connected(self) -> bool:
        return self._connected and self.vr_monitor is not None and self.vr_thread is not None and self.vr_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("XLerobot VR is already connected.")
        if not VR_AVAILABLE:
            raise RuntimeError("VR Monitor is not available. Please check VR runtime.")

        self.vr_monitor = VRMonitor()
        if not self.vr_monitor.initialize():
            raise RuntimeError("VR monitor initialization failed")

        self.vr_thread = threading.Thread(
            target=lambda: asyncio.run(self.vr_monitor.start_monitoring()),
            daemon=True,
        )
        self.vr_thread.start()

        wait_start = time.time()
        while time.time() - wait_start < self.config.vr_connection_timeout:
            goals = self.vr_monitor.get_latest_goal_nowait()
            if goals and any(
                goal is not None and getattr(goal, "target_position", None) is not None
                for goal in (goals.get("left"), goals.get("right"))
            ):
                self._connected = True
                break
            time.sleep(0.1)

        if not self._connected:
            raise RuntimeError("VR client connection timeout. Please open the VR page first.")

        if calibrate and self.latest_robot_obs is not None:
            self.calibrate()

    def calibrate(self) -> None:
        if self.latest_robot_obs is None:
            raise RuntimeError("No robot observation available yet. Call send_feedback() before calibration.")

        self.left_arm = None
        self.right_arm = None

        controlled_joint_map, controlled_prefix = self._resolve_controlled_arm()
        self.primary_arm = SimpleTeleopArm(
            controlled_joint_map,
            self.latest_robot_obs,
            SO101Kinematics(),
            prefix=controlled_prefix,
            config=self.config,
        )

        has_dual_arm_obs = any(key.startswith("left_arm_") for key in self.latest_robot_obs) and any(
            key.startswith("right_arm_") for key in self.latest_robot_obs
        )
        if has_dual_arm_obs and not self.config.single_arm_mode:
            left_joint_map = resolve_arm_joint_map(self.latest_robot_obs, "left")
            right_joint_map = resolve_arm_joint_map(self.latest_robot_obs, "right")
            self.left_arm = SimpleTeleopArm(
                left_joint_map,
                self.latest_robot_obs,
                SO101Kinematics(),
                prefix="left",
                config=self.config,
            )
            self.right_arm = SimpleTeleopArm(
                right_joint_map,
                self.latest_robot_obs,
                SO101Kinematics(),
                prefix="right",
                config=self.config,
            )
        elif controlled_prefix == "left":
            self.left_arm = self.primary_arm
        else:
            self.right_arm = self.primary_arm
        if (
            self.config.enable_head
            and "head_motor_1.pos" in self.latest_robot_obs
            and "head_motor_2.pos" in self.latest_robot_obs
        ):
            self.head_control = SimpleHeadControl(self.latest_robot_obs, config=self.config)
        else:
            self.head_control = None
        self.base_control = SimpleBaseControl(self.latest_robot_obs, config=self.config)
        self._calibrated = True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self.latest_robot_obs = feedback
        if not self._calibrated:
            self.calibrate()

    def _read_goals(self):
        if self.vr_monitor is None:
            return None
        try:
            return self.vr_monitor.get_latest_goal_nowait()
        except Exception as exc:
            logger.debug(f"Failed to read VR goals: {exc}")
            return None

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()
        if not self.is_connected or self.latest_robot_obs is None:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return {}

        if not self._calibrated:
            self.calibrate()

        goals = self._read_goals()
        if not goals:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return {}

        left_goal = goals.get("left")
        right_goal = goals.get("right")
        controller_side = (self.config.arm_controller or "right").lower()
        controller_goal = left_goal if controller_side == "left" else right_goal

        # Event source selection:
        # - Dual-arm mode keeps the historical behavior (left controller drives events).
        # - Single-arm mode prioritizes the active controller, then falls back to left.
        event_goal = None
        if self.config.single_arm_mode:
            if controller_goal is not None and hasattr(controller_goal, "metadata"):
                event_goal = controller_goal
            elif left_goal is not None and hasattr(left_goal, "metadata"):
                event_goal = left_goal
        elif left_goal is not None and hasattr(left_goal, "metadata"):
            event_goal = left_goal

        if event_goal is not None:
            self.vr_event_handler.process_left_controller(event_goal.metadata)

        if left_goal is not None and hasattr(left_goal, "metadata"):
            if self.config.enable_left_hand and self.left_arm is not None and not self.config.single_arm_mode:
                self.left_arm.handle_vr_input(left_goal)
        if controller_goal is not None:
            if self.config.single_arm_mode:
                if self.primary_arm is not None:
                    self.primary_arm.handle_vr_input(controller_goal)
            elif controller_side == "right" and self.right_arm is not None:
                self.right_arm.handle_vr_input(right_goal)
            if self.head_control is not None and right_goal is not None:
                self.head_control.handle_vr_input(right_goal)
            if self.base_control is not None and right_goal is not None:
                self.base_control.handle_vr_input(right_goal)

        action = {}
        if self.left_arm is not None and self.config.enable_left_hand:
            action.update(self.left_arm.p_control_action(self.latest_robot_obs))
        if self.config.single_arm_mode and self.primary_arm is not None and self.primary_arm is not self.left_arm:
            action.update(self.primary_arm.p_control_action(self.latest_robot_obs))
        if self.right_arm is not None and (not self.config.single_arm_mode or self.right_arm is not self.primary_arm):
            action.update(self.right_arm.p_control_action(self.latest_robot_obs))
        if self.head_control is not None:
            action.update(self.head_control.p_control_action(self.latest_robot_obs))
        if self.base_control is not None:
            action.update(self.base_control.p_control_action(self.latest_robot_obs))

        if self.vr_event_handler.get_events().get("reset_position", False):
            if self.left_arm is not None and self.config.enable_left_hand:
                action.update(self.left_arm.move_to_zero_position())
            if self.config.single_arm_mode and self.primary_arm is not None and self.primary_arm is not self.left_arm:
                action.update(self.primary_arm.move_to_zero_position())
            if self.right_arm is not None and (not self.config.single_arm_mode or self.right_arm is not self.primary_arm):
                action.update(self.right_arm.move_to_zero_position())
            if self.head_control is not None:
                action.update(self.head_control.move_to_zero_position())
            if self.base_control is not None:
                action.update(self.base_control.move_to_zero_velocity())

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        return action

    def get_teleop_events(self) -> dict[str, bool]:
        events = self.vr_event_handler.get_events()
        self.vr_event_handler.reset_latched_events()
        return events

    def send_feedback_wait(self) -> None:
        return None

    def disconnect(self) -> None:
        self._connected = False
        self._calibrated = False
        logger.info("[VR] Disconnected")
