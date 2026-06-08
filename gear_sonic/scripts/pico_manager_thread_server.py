# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python pico_manager_thread_server.py --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python pico_manager_thread_server.py --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python pico_manager_thread_server.py --vr3pt_live

# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
    #   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

"""

from collections import defaultdict, deque
from enum import Enum, IntEnum
import os
import subprocess
import threading
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import yaml
import zmq

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None


class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


class WBCDCompetitionMode(Enum):
    TRANSPORT = 0
    STAND_MANIP = 1
    HALF_SQUAT_MANIP = 2
    STAND_RECOVERY = 3
    KNEEL_MANIP = 4


class WBCDKneelRecoveryStage(Enum):
    NONE = 0
    KNEEL_HOLD = 1
    ONE_KNEEL_HOLD = 2
    SQUAT_HOLD = 3
    SQUAT_TO_STAND = 4


class WBCDCompetitionConfig:
    """Small hot-reloadable YAML config for competition-only Pico controls."""

    DEFAULTS = {
        "enabled": False,
        "reload_interval_sec": 1.0,
        "height": {
            "stand_manip": -1.0,
            "half_squat_default": 0.55,
            "half_squat_min": 0.45,
            "half_squat_max": 0.62,
            "kneel_default": 0.50,
            "kneel_min": 0.30,
            "kneel_max": 0.70,
            "kneel_adjust_step_m": 0.02,
            "adjust_speed_mps": 0.06,
            "stand_recovery_target": 0.74,
            "stand_recovery_speed_mps": 0.10,
        },
        "entry": {
            "require_robot_feedback": True,
            "hold_before_squat_sec": 0.30,
            "squat_enter_speed_mps": 0.08,
            "print_vr_target_debug": False,
        },
        "recovery": {
            "kneel_hold_sec": 0.30,
            "one_kneel_height": 0.50,
            "one_kneel_hold_sec": 0.60,
            "squat_recovery_height": 0.58,
            "squat_hold_sec": 0.50,
            "squat_to_stand_speed_mps": 0.06,
            "allow_recovery_motion": False,
        },
        "lean": {
            "enabled": True,
            "default_pitch_deg": 0.0,
            "min_pitch_deg": 0.0,
            "max_pitch_deg": 18.0,
            "pitch_step_deg": 3.0,
            "torso_forward_offset_per_deg": 0.003,
            "torso_down_offset_per_deg": 0.0015,
        },
        "movement": {
            "joystick_dead_zone": 0.10,
            "stand_manip_max_vx": 0.10,
            "stand_manip_max_wz": 0.25,
            "stand_recovery_max_vx": 0.08,
            "stand_recovery_max_wz": 0.25,
            "half_squat_allow_motion": False,
        },
        "buttons": {
            "modifier": "left_menu_button",
            "stand_manip": "A",
            "half_squat_manip": "X",
            "kneel_manip": "Y",
            "hold_stand_recovery": "B",
            "height_up": "Y",
            "height_down": "X",
            "lean_forward": "B",
            "lean_back": "A",
        },
        "logging": {
            "print_mode_changes": True,
            "print_config_reload": True,
        },
    }

    def __init__(self, path: str | None = None):
        self.path = path or default_wbcd_config_path()
        self.data = self._deepcopy_defaults()
        self._mtime = None
        self._last_check = 0.0
        self.load(force=True)

    @staticmethod
    def _deepcopy_defaults():
        import copy

        return copy.deepcopy(WBCDCompetitionConfig.DEFAULTS)

    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = WBCDCompetitionConfig._merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def load(self, force: bool = False) -> bool:
        try:
            if not os.path.exists(self.path):
                if force:
                    print(f"[WBCD] Config not found, using defaults: {self.path}")
                return False
            mtime = os.path.getmtime(self.path)
            if not force and self._mtime == mtime:
                return False
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("top-level YAML value must be a mapping")
            self.data = self._merge(self._deepcopy_defaults(), loaded)
            self._mtime = mtime
            if self.get_bool("logging", "print_config_reload", default=True):
                print(f"[WBCD] Config loaded: {self.path}")
            return True
        except Exception as e:
            print(f"[WBCD] Config reload failed, keeping previous config: {e}")
            return False

    def reload_if_needed(self):
        now = time.time()
        interval = self.get_float("reload_interval_sec", default=1.0)
        if now - self._last_check >= interval:
            self._last_check = now
            self.load(force=False)

    def get(self, *keys, default=None):
        value = self.data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def get_float(self, *keys, default: float = 0.0) -> float:
        try:
            return float(self.get(*keys, default=default))
        except (TypeError, ValueError):
            return float(default)

    def get_bool(self, *keys, default: bool = False) -> bool:
        value = self.get(*keys, default=default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def get_str(self, *keys, default: str = "") -> str:
        value = self.get(*keys, default=default)
        return str(value)

    @property
    def enabled(self) -> bool:
        return self.get_bool("enabled", default=False)


def default_wbcd_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", "config", "wbcd_pico_competition.yaml"))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _face_button_pressed(name: str, a_pressed: bool, b_pressed: bool, x_pressed: bool, y_pressed: bool) -> bool:
    name = name.upper()
    if name == "A":
        return bool(a_pressed)
    if name == "B":
        return bool(b_pressed)
    if name == "X":
        return bool(x_pressed)
    if name == "Y":
        return bool(y_pressed)
    return False


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]


# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    """
    Generate finger position data from Pico controller button states.

    Args:
        hand: "left" or "right"
        trigger: Trigger button value (0-1)
        grip: Grip button value (0-1)

    Returns:
        Array of shape [25, 4, 4] representing fingertip positions
    """
    fingertips = np.zeros([25, 4, 4])

    thumb = 0
    middle = 10
    # Control thumb based on shoulder button state (index 4 is thumb tip)
    fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
    if trigger > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0  # close middle

    return fingertips


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


def compute_from_body_poses(parent_indices: list, device, body_poses_np: np.ndarray):
    """
    Compute local joints and body orientation from provided body_poses_np.
    """
    positions = body_poses_np[:, :3]
    global_quats = body_poses_np[:, [6, 3, 4, 5]]

    # Convert to local rotations
    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)


def init_hand_ik_solvers():
    """Initialize hand IK solvers if available."""
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes():
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons():
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks():
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons():
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right hand joints using IK solvers, or zeros if unavailable."""
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolate two quaternions and renormalize. Input shape (4,), xyzw order.
    Ensures shortest path by flipping sign if dot < 0.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Interpolate axis-angle joint poses by converting to quats, lerp-normalize, then back.
    prev_pose, curr_pose: (21,3) axis-angle (rotvec)
    Returns (21,3) axis-angle.
    """
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()  # (N,4) xyzw
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


def _pose_stream_common(
    socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Shared pose streaming loop used by run_pico."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
        )

    # Create reader and start it
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
    )

    streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix=log_prefix,
    )

    if stop_event is None:
        stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            streamer.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()


class ThreePointPose:
    """
    Encapsulates everything around calculating 3-point pose from SMPL input.

    This includes:
    - Processing SMPL poses to extract 3-point VR pose (L-Wrist, R-Wrist, Neck)
    - Calibration logic to align VR poses with G1 robot
    - Optional visualization of 3-point poses

    Calibration is done in two steps:
    1. Neck orientation: Captures initial neck orientation to align subsequent poses as upright
    2. Wrist positions: Aligns wrist positions to match G1 robot key frame positions
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        # Robot model for FK-based calibration (headless, no display required)
        self._robot_model = robot_model
        if self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for FK calibration")

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        # Calibration state — triggered explicitly by calibrate_now() or reset_with_measured_q()
        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None  # inv(initial neck quat)
        self._calibration_lwrist_offset: np.ndarray | None = None  # position offset
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None  # orientation offset
        self._calibration_rwrist_rot_offset: sRot | None = None
        # Override robot q for FK during recalibration (e.g. measured joints for VR 3PT)
        self._override_robot_q: np.ndarray | None = None

    @property
    def is_pending(self) -> bool:
        """Check if calibration is pending."""
        return self._calibration_pending

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been captured."""
        return self._calibration_neck_quat_inv is not None

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        # Extract raw 3-point pose from SMPL
        vr_3pt_pose_raw = _process_3pt_pose(smpl_pose_np)

        # Capture calibration on first valid frame (or after reset)
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        # Apply calibration to get the final pose
        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate using current SMPL frame against FK of all-zero body joints.
        Operator should be in zero-reference pose when calling this."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (zero-pose reference)")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def calibrate_with_measured_q_now(
        self,
        body_poses_np: np.ndarray,
        body_q_measured: np.ndarray,
    ) -> bool:
        """Calibrate wrist offsets against the robot's current measured pose.

        WBCD manipulation mode uses this to make the operator's current safe
        hand pose map to the robot's current hand pose before switching to
        VR_3PT control.
        """
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._calibration_lwrist_offset = None
            self._calibration_rwrist_offset = None
            self._calibration_lwrist_rot_offset = None
            self._calibration_rwrist_rot_offset = None
            self._override_robot_q = body_q_measured.copy()
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] WBCD continuous VR 3PT calibration completed")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] WBCD continuous calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture calibration offsets from vr_3pt_pose against G1 FK reference.
        If neck calibration already exists (e.g. from calibrate_now), it is preserved
        to avoid jumps from SMPL noise during recalibration."""

        # Step 1: Neck orientation — only capture if not already set
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Step 2: Rotate VR wrist positions/orientations by neck inverse
        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)

        # Step 3: Get G1 FK reference poses
        if self._robot_model is None:
            raise RuntimeError(
                "Robot model is required for calibration but was not loaded. "
                "Ensure the G1 robot model and URDF are available."
            )
        if get_g1_key_frame_poses is None:
            raise RuntimeError(
                "get_g1_key_frame_poses could not be imported. "
                "Ensure gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer is available."
            )

        # Convert 29-DOF override to full model config if needed
        if self._override_robot_q is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=self._override_robot_q[:29]
            )
        else:
            robot_q = None
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)

        g1_lwrist_pos = g1_poses["left_wrist"]["position"]
        g1_rwrist_pos = g1_poses["right_wrist"]["position"]
        g1_lwrist_rot = sRot.from_quat(
            g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        g1_rwrist_rot = sRot.from_quat(
            g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        # Compute position offsets: calibrated = neck_corrected - offset
        self._calibration_lwrist_offset = lwrist_pos_corrected - g1_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - g1_rwrist_pos

        # Compute orientation offsets: calibrated = rot_offset * neck_corrected
        self._calibration_lwrist_rot_offset = g1_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = g1_rwrist_rot * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        self._override_robot_q = None

        # Log summary
        source = "override q" if g1_lwrist_pos.any() else "default/zero"
        print(
            f"[{self.log_prefix}] Calibration captured (FK ref: {source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        """Apply stored calibration offsets to raw VR 3-point pose."""
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Neck orientation: calibrated = inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        # Wrist orientations: rot_offset * (neck_inv * current)
        if self._calibration_lwrist_rot_offset is not None:
            lw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        # Neck position via kinematic chain: root → torso_link (+Z) → neck (along calibrated Z)
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        calibrated[2, :3] = (
            np.array([0, 0, self.TORSO_LINK_OFFSET_Z]) + self.NECK_LINK_LENGTH * neck_z
        ).astype(np.float32)

        return calibrated

    def _clear_calibration(self):
        """Clear all calibration state."""
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = None

    def reset(self) -> None:
        """Reset calibration. Next process_smpl_pose() call will recalibrate."""
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Recalibrate wrist offsets using measured robot joints (29 DOFs).
        Preserves neck calibration to avoid jumps from SMPL noise.
        Next process_smpl_pose() will recompute wrist offsets against FK of these joints."""
        # Preserve neck calibration — only clear wrist offsets
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")


class PoseStreamer:
    """Encapsulates the pose streaming loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        log_prefix: str = "PoseLoop",
    ):
        self.socket = socket
        self.reader = reader
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix

        # Injected dependencies
        self.reader = reader
        self.three_point = three_point

        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self.parent_indices = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
        ][:24]

        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        # NOTE: Sleep budget set to 95% of the ideal frame period so that the actual
        # FPS lands closer to target_fps despite per-frame processing overhead.
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.frame_start = time.time()

        # Data collection button state tracking (edge-triggered)
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False

        self.buffer_cleared = (
            True  # Start with buffer cleared - wait for full buffer before first send
        )
        self.yaw_accumulator = YawAccumulator()

    def reset_yaw(self):
        """Called when entering pose mode. Resets yaw only.
        Calibration is triggered separately by the operator (A+B+X+Y → calibrate_now)."""
        self.yaw_accumulator.reset()

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self):
        """Execute one iteration of the pose streaming loop."""
        sample = self.reader.get_latest()

        if sample is None:
            time.sleep(0.005)
            return

        latest_data = compute_from_body_poses(
            self.parent_indices, self.device, sample["body_poses_np"]
        )
        (left_menu_button, left_trigger, right_trigger, left_grip, right_grip) = (
            get_controller_inputs()
        )
        # Get A and B button states for data collection control
        a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()

        # Data collection toggle logic (edge-triggered)
        # Left grip + A = toggle_data_collection
        # Left grip + B = toggle_data_abort
        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5

        # Detect rising edge
        toggle_data_collection = toggle_data_collection_tmp and not self.toggle_data_collection_last
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        left_hand_joints, right_hand_joints = compute_hand_joints_from_inputs(
            self.left_hand_ik_solver,
            self.right_hand_ik_solver,
            left_trigger,
            left_grip,
            right_trigger,
            right_grip,
        )
        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        )
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return
        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = _interp_pose_axis_angle(self.prev_smpl_pose_np, smpl_pose_np, alpha).astype(
            np.float32
        )
        use_body_quat = _quat_lerp_normalized(self.prev_body_quat_np, body_quat_np, alpha).astype(
            np.float32
        )
        N = len(self.frame_buffer["frame_index"])

        ##### From @Jiefeng for directly setting the joint position ######
        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        # G1_L_ELBOW_IDX = 0
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27

        # G1_R_ELBOW_IDX = 0
        G1_R_WRIST_ROLL_IDX = 24  # Done
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28
        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(
            smpl_l_elbow_aa, g1_l_elbow_axis
        )

        g1_r_elbow_axis = np.array([0, 1, 0])
        g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(
            smpl_r_elbow_aa, g1_r_elbow_axis
        )

        # Move elbow roll/yaw into wrist while preserving wrist pitch from SMPL
        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
        g1_l_wrist_pitch = -l_wrist_euler[:, 1]
        g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

        g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch = -r_wrist_euler[:, 1]
        g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

        joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

        # Process SMPL pose to get calibrated 3-point VR pose and update visualization
        # Pass SMPL local joints for optional body visualization in the VR3Pt viewer
        smpl_joints_for_vis = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            if self.three_point.enable_smpl_vis
            else None
        )
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )
        ##### From @Jiefeng for directly setting the joint position ######

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))
        N = len(self.frame_buffer["frame_index"])

        # Wait for buffer to be completely filled before sending first message after clearing
        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            # Buffer is now full with fresh data, can start sending
            self.buffer_cleared = False

        # Get joystick axes for yaw accumulation
        _, _, rx, _ = get_controller_axes()
        self.yaw_accumulator.update(rx, self.frame_time)

        # Only send if buffer is full and we're not waiting for fresh data
        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "frame_index": np.array((self.frame_buffer["frame_index"]), dtype=np.int64),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_grip": np.array([left_grip], dtype=np.float32),
                "right_grip": np.array([right_grip], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
                "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
                "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }

            packed_message = pack_pose_message(numpy_data, topic="pose")
            self.socket.send(packed_message)

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Run Pico body tracking with real-time visualization and ZMQ streaming."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
        )
    finally:
        socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")


class FeedbackReader:
    """Reads feedback from robot via ZMQ and processes measured upper body position to use as frozen targets."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")

        self.upper_body_joint_indices = self._get_upper_body_joint_indices()

        self.upper_body_position_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        # Full body joint configuration (29 DOFs) as measured from robot,
        # used for FK when recalibrating VR 3PT tracking against actual robot pose
        self.full_body_q_measured: np.ndarray | None = None

    def _get_upper_body_joint_indices(self) -> list[int]:
        # TODO: get from robot model, not hardcoded
        # robot_model = instantiate_g1_robot_model()
        # return robot_model.get_joint_group_indices("upper_body")
        return [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

    def poll_feedback(self):
        """Poll for feedback once, and update internal state."""
        (
            self.upper_body_position_target,
            self.left_hand_position_target,
            self.right_hand_position_target,
            self.full_body_q_measured,
        ) = self._process_upper_body_position_targets()
        print("[PlannerLoop] Saved upper body position target:", self.upper_body_position_target)

    def _process_upper_body_position_targets(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        data = self.poller.get_data()

        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None, None, None, None

        unpacked = msgpack.unpackb(data, raw=False)
        full_body_q = None
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            body_q = [body_q_swizzled[i] for i in self.upper_body_joint_indices]
        else:
            print("[PlannerLoop] body_q_measured not in feedback data")
            body_q = None

        if "left_hand_q_measured" in unpacked:
            left_hand_q = unpacked["left_hand_q_measured"]
        else:
            print("[PlannerLoop] left_hand_q_measured not in feedback data")
            left_hand_q = None

        if "right_hand_q_measured" in unpacked:
            right_hand_q = unpacked["right_hand_q_measured"]
        else:
            print("[PlannerLoop] right_hand_q_measured not in feedback data")
            right_hand_q = None

        return body_q, left_hand_q, right_hand_q, full_body_q


class PlannerStreamer:
    """Encapsulates the planner control loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        poll_hz: int = 20,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
        wbcd_config: WBCDCompetitionConfig | None = None,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        # Current locomotion mode, default IDLE
        self.mode = LocomotionMode.IDLE
        self.prev_ab = False
        self.prev_xy = False
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator()
        self.last_send = time.time()
        self.last_xrt_timestamp = None
        self.wbcd_config = wbcd_config
        self.wbcd_mode = WBCDCompetitionMode.TRANSPORT
        self.wbcd_height = self._cfg_float("height", "half_squat_default", default=0.55)
        self.wbcd_recovery_hold = False
        self._wbcd_return_to_pose = False
        self._wbcd_entry_hold_until = 0.0
        self._wbcd_squat_target_height = self.wbcd_height
        self._prev_wbcd_height_up = False
        self._prev_wbcd_height_down = False
        self._kneel_recovery_stage = WBCDKneelRecoveryStage.NONE
        self._kneel_recovery_stage_elapsed = 0.0
        self.wbcd_lean_pitch_deg = self._cfg_float("lean", "default_pitch_deg", default=0.0)
        self._prev_wbcd_lean_forward = False
        self._prev_wbcd_lean_back = False

        # Hand IK solvers for trigger-controlled hand open/close in VR 3PT mode
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()

    def _cfg_float(self, *keys, default: float = 0.0) -> float:
        if self.wbcd_config is None:
            return float(default)
        return self.wbcd_config.get_float(*keys, default=default)

    def _cfg_bool(self, *keys, default: bool = False) -> bool:
        if self.wbcd_config is None:
            return bool(default)
        return self.wbcd_config.get_bool(*keys, default=default)

    def wbcd_enabled(self) -> bool:
        return self.wbcd_config is not None and self.wbcd_config.enabled

    def is_wbcd_active(self) -> bool:
        return self.wbcd_mode != WBCDCompetitionMode.TRANSPORT

    def _clamp_wbcd_lean(self):
        min_pitch = self._cfg_float("lean", "min_pitch_deg", default=0.0)
        max_pitch = self._cfg_float("lean", "max_pitch_deg", default=18.0)
        self.wbcd_lean_pitch_deg = clamp(self.wbcd_lean_pitch_deg, min_pitch, max_pitch)

    def _reset_wbcd_lean(self):
        self.wbcd_lean_pitch_deg = self._cfg_float("lean", "default_pitch_deg", default=0.0)
        self._clamp_wbcd_lean()

    def _wbcd_lean_offsets(self) -> tuple[float, float]:
        forward_per_deg = self._cfg_float("lean", "torso_forward_offset_per_deg", default=0.003)
        down_per_deg = self._cfg_float("lean", "torso_down_offset_per_deg", default=0.0015)
        return forward_per_deg * self.wbcd_lean_pitch_deg, down_per_deg * self.wbcd_lean_pitch_deg

    def _print_wbcd_lean_state(self):
        forward_offset, down_offset = self._wbcd_lean_offsets()
        print(
            "[WBCD] Lean pitch -> "
            f"{self.wbcd_lean_pitch_deg:.1f} deg, "
            f"torso_forward=+{forward_offset:.3f} m, "
            f"torso_down=-{down_offset:.3f} m"
        )

    def prepare_wbcd_vr3pt_entry(self, mode: WBCDCompetitionMode) -> bool:
        sample = self.reader.get_latest()
        if sample is None:
            print(f"[WBCD] Cannot enter {mode.name}: no Pico SMPL sample")
            return False

        self.feedback_reader.poll_feedback()
        if self.feedback_reader.full_body_q_measured is None:
            if self._cfg_bool("entry", "require_robot_feedback", default=True):
                print(f"[WBCD] Cannot enter {mode.name}: no robot feedback full_body_q_measured")
                return False
            print(f"[WBCD] Entering {mode.name} without robot feedback; continuous calibration skipped")
            return True

        ok = self.three_point.calibrate_with_measured_q_now(
            sample["body_poses_np"],
            self.feedback_reader.full_body_q_measured,
        )
        if not ok:
            print(f"[WBCD] Cannot enter {mode.name}: continuous VR 3PT calibration failed")
            return False
        return True

    def enter_wbcd_mode(self, mode: WBCDCompetitionMode):
        self._prev_wbcd_height_up = False
        self._prev_wbcd_height_down = False
        self._prev_wbcd_lean_forward = False
        self._prev_wbcd_lean_back = False
        if mode != WBCDCompetitionMode.STAND_RECOVERY:
            self._kneel_recovery_stage = WBCDKneelRecoveryStage.NONE
            self._kneel_recovery_stage_elapsed = 0.0
        if mode == WBCDCompetitionMode.HALF_SQUAT_MANIP:
            default_height = self._cfg_float("height", "half_squat_default", default=0.55)
            min_height = self._cfg_float("height", "half_squat_min", default=0.45)
            max_height = self._cfg_float("height", "half_squat_max", default=0.62)
            self._wbcd_squat_target_height = clamp(default_height, min_height, max_height)
            if self.wbcd_mode == WBCDCompetitionMode.TRANSPORT:
                hold_sec = self._cfg_float("entry", "hold_before_squat_sec", default=0.30)
                self._wbcd_entry_hold_until = time.time() + max(0.0, hold_sec)
                self.wbcd_height = self._cfg_float("height", "stand_recovery_target", default=0.74)
            else:
                self._wbcd_entry_hold_until = 0.0
                self.wbcd_height = self._wbcd_squat_target_height
        elif mode == WBCDCompetitionMode.KNEEL_MANIP:
            default_height = self._cfg_float("height", "kneel_default", default=0.50)
            min_height = self._cfg_float("height", "kneel_min", default=0.30)
            max_height = self._cfg_float("height", "kneel_max", default=0.70)
            self.wbcd_height = clamp(default_height, min_height, max_height)
            self._wbcd_entry_hold_until = 0.0
        elif mode == WBCDCompetitionMode.STAND_MANIP:
            self.wbcd_height = self._cfg_float("height", "stand_manip", default=-1.0)
            self._wbcd_entry_hold_until = 0.0
        elif mode == WBCDCompetitionMode.STAND_RECOVERY:
            if self.wbcd_mode == WBCDCompetitionMode.STAND_MANIP:
                self.wbcd_height = self._cfg_float("height", "stand_recovery_target", default=0.74)
            elif self.wbcd_height < 0.0:
                self.wbcd_height = self._cfg_float("height", "half_squat_default", default=0.55)
            self._wbcd_entry_hold_until = 0.0
        self.wbcd_mode = mode
        self._wbcd_return_to_pose = False
        if self._cfg_bool("logging", "print_mode_changes", default=True):
            print(f"[WBCD] Mode -> {self.wbcd_mode.name}, height={self.wbcd_height:.3f}")

    def clear_wbcd_mode(self):
        if self.wbcd_mode != WBCDCompetitionMode.TRANSPORT:
            self.wbcd_mode = WBCDCompetitionMode.TRANSPORT
            self.wbcd_recovery_hold = False
            self._wbcd_return_to_pose = False
            self._wbcd_entry_hold_until = 0.0
            self._prev_wbcd_height_up = False
            self._prev_wbcd_height_down = False
            self._prev_wbcd_lean_forward = False
            self._prev_wbcd_lean_back = False
            self._kneel_recovery_stage = WBCDKneelRecoveryStage.NONE
            self._kneel_recovery_stage_elapsed = 0.0
            self._reset_wbcd_lean()
            if self._cfg_bool("logging", "print_mode_changes", default=True):
                print("[WBCD] Mode -> TRANSPORT")

    def set_wbcd_recovery_hold(self, held: bool):
        self.wbcd_recovery_hold = bool(held)

    def start_kneel_recovery(self):
        self.wbcd_mode = WBCDCompetitionMode.STAND_RECOVERY
        self.wbcd_recovery_hold = True
        self._wbcd_return_to_pose = False
        self._wbcd_entry_hold_until = 0.0
        self._kneel_recovery_stage = WBCDKneelRecoveryStage.KNEEL_HOLD
        self._kneel_recovery_stage_elapsed = 0.0
        if self._cfg_bool("logging", "print_mode_changes", default=True):
            print(
                "[WBCD] Mode -> STAND_RECOVERY, "
                f"kneel_stage={self._kneel_recovery_stage.name}, height={self.wbcd_height:.3f}"
            )

    def consume_wbcd_return_to_pose(self) -> bool:
        value = self._wbcd_return_to_pose
        self._wbcd_return_to_pose = False
        return value

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def save_upper_body_position_target(self):
        """Poll feedback and save upper body position target."""
        self.feedback_reader.poll_feedback()

    def recalibrate_for_vr3pt(self):
        """
        Recalibrate VR 3-point pose tracking using the robot's current measured joints.

        Polls the g1_debug feedback to get the robot's actual joint state, then
        schedules recalibration so VR tracking aligns with the robot's current pose.
        This prevents sudden jumps when entering VR 3PT mode from PLANNER mode.
        """
        self.feedback_reader.poll_feedback()
        if self.feedback_reader.full_body_q_measured is not None:
            self.three_point.reset_with_measured_q(self.feedback_reader.full_body_q_measured)
            print("[PlannerLoop] VR 3PT recalibration scheduled with measured robot pose")
        else:
            # Fallback: use zeros if no feedback available
            print(
                "[PlannerLoop] WARNING: No feedback data for VR 3PT recalibration, "
                "using zero body_q as fallback"
            )
            self.three_point.reset_with_measured_q(np.zeros(29, dtype=np.float64))

    def _compute_wbcd_movement(self, max_vx: float, max_wz: float) -> tuple[list[float], list[float], float, LocomotionMode]:
        lx, ly, rx, _ = get_controller_axes()
        dead_zone = self._cfg_float("movement", "joystick_dead_zone", default=0.10)
        self.yaw_accumulator.dyaw = max_wz * (-rx) * self.dt
        if abs(rx) >= dead_zone:
            self.yaw_accumulator.yaw_angle_rad += self.yaw_accumulator.dyaw
            self.yaw_accumulator.heading = [
                np.cos(self.yaw_accumulator.yaw_angle_rad),
                np.sin(self.yaw_accumulator.yaw_angle_rad),
                0.0,
            ]
        facing = self.yaw_accumulator.heading

        raw = abs(ly)
        if raw < dead_zone:
            return [0.0, 0.0, 0.0], facing, -1.0, LocomotionMode.IDLE

        mag = (raw - dead_zone) / max(1e-6, 1.0 - dead_zone)
        mag = clamp(mag, 0.0, 1.0)
        direction_sign = 1.0 if ly >= 0.0 else -1.0
        movement_local = np.array([0.0, direction_sign * mag])
        perp_x, perp_y = -facing[1], facing[0]
        rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
        movement_global = rotation_facing @ movement_local
        speed = max_vx * mag
        return [movement_global[0], movement_global[1], 0.0], facing, speed, LocomotionMode.SLOW_WALK

    def _collect_vr3pt_targets(self):
        sample = self.reader.get_latest()
        if sample is None:
            return None

        print("[PlannerLoop] Sending VR 3-point pose as target")
        vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
        if self._cfg_bool("lean", "enabled", default=True):
            self._clamp_wbcd_lean()
            pitch_deg = self.wbcd_lean_pitch_deg
            if abs(pitch_deg) > 1e-6:
                torso_idx = 2
                forward_offset, down_offset = self._wbcd_lean_offsets()
                vr_3pt_pose[torso_idx, 0] += forward_offset
                vr_3pt_pose[torso_idx, 2] -= down_offset
                torso_rot = sRot.from_quat(vr_3pt_pose[torso_idx, 3:], scalar_first=True)
                lean_rot = sRot.from_euler("y", -pitch_deg, degrees=True)
                vr_3pt_pose[torso_idx, 3:] = (lean_rot * torso_rot).as_quat(scalar_first=True)
        vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
        vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

        (
            _left_menu_button,
            left_trigger,
            right_trigger,
            left_grip,
            right_grip,
        ) = get_controller_inputs()
        lh_joints, rh_joints = compute_hand_joints_from_inputs(
            self.left_hand_ik_solver,
            self.right_hand_ik_solver,
            left_trigger,
            left_grip,
            right_trigger,
            right_grip,
        )
        left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
        right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()
        return vr_3pt_position, vr_3pt_orientation, left_hand_position, right_hand_position

    def _run_wbcd_once(self, a_pressed: bool, b_pressed: bool, x_pressed: bool, y_pressed: bool):
        self.wbcd_config.reload_if_needed()
        left_menu_button, _, _, _, _ = get_controller_inputs()
        if self._cfg_bool("lean", "enabled", default=True):
            lean_forward_now = (
                not left_menu_button
                and _face_button_pressed(
                    self.wbcd_config.get_str("buttons", "lean_forward", default="B"),
                    a_pressed,
                    b_pressed,
                    x_pressed,
                    y_pressed,
                )
            )
            lean_back_now = (
                not left_menu_button
                and _face_button_pressed(
                    self.wbcd_config.get_str("buttons", "lean_back", default="A"),
                    a_pressed,
                    b_pressed,
                    x_pressed,
                    y_pressed,
                )
            )
            step = self._cfg_float("lean", "pitch_step_deg", default=3.0)
            if lean_forward_now and not self._prev_wbcd_lean_forward:
                self.wbcd_lean_pitch_deg += step
                self._clamp_wbcd_lean()
                self._print_wbcd_lean_state()
            if lean_back_now and not self._prev_wbcd_lean_back:
                self.wbcd_lean_pitch_deg -= step
                self._clamp_wbcd_lean()
                self._print_wbcd_lean_state()
            self._prev_wbcd_lean_forward = lean_forward_now
            self._prev_wbcd_lean_back = lean_back_now

        mode_to_send = LocomotionMode.IDLE
        height = -1.0
        movement = [0.0, 0.0, 0.0]
        facing = self.yaw_accumulator.update(0.0, self.dt)
        speed = -1.0

        if self.wbcd_mode == WBCDCompetitionMode.STAND_MANIP:
            height = self._cfg_float("height", "stand_manip", default=-1.0)
            max_vx = self._cfg_float("movement", "stand_manip_max_vx", default=0.10)
            max_wz = self._cfg_float("movement", "stand_manip_max_wz", default=0.25)
            movement, facing, speed, mode_to_send = self._compute_wbcd_movement(max_vx, max_wz)

        elif self.wbcd_mode == WBCDCompetitionMode.HALF_SQUAT_MANIP:
            min_height = self._cfg_float("height", "half_squat_min", default=0.45)
            max_height = self._cfg_float("height", "half_squat_max", default=0.62)
            entry_hold_active = time.time() < self._wbcd_entry_hold_until
            entry_lowering_active = (
                not entry_hold_active and self.wbcd_height > self._wbcd_squat_target_height
            )
            if not left_menu_button and not entry_hold_active and not entry_lowering_active:
                adjust_speed = self._cfg_float("height", "adjust_speed_mps", default=0.06)
                if _face_button_pressed(
                    self.wbcd_config.get_str("buttons", "height_up", default="Y"),
                    a_pressed,
                    b_pressed,
                    x_pressed,
                    y_pressed,
                ):
                    self.wbcd_height += adjust_speed * self.dt
                if _face_button_pressed(
                    self.wbcd_config.get_str("buttons", "height_down", default="X"),
                    a_pressed,
                    b_pressed,
                    x_pressed,
                    y_pressed,
                ):
                    self.wbcd_height -= adjust_speed * self.dt
            if entry_hold_active:
                self.wbcd_height = self._cfg_float("height", "stand_recovery_target", default=0.74)
            elif entry_lowering_active:
                enter_speed = self._cfg_float("entry", "squat_enter_speed_mps", default=0.08)
                self.wbcd_height = max(
                    self._wbcd_squat_target_height,
                    self.wbcd_height - max(0.0, enter_speed) * self.dt,
                )
            else:
                self.wbcd_height = clamp(self.wbcd_height, min_height, max_height)
            height = self.wbcd_height
            mode_to_send = LocomotionMode.IDLE_SQUAT

        elif self.wbcd_mode == WBCDCompetitionMode.KNEEL_MANIP:
            min_height = self._cfg_float("height", "kneel_min", default=0.30)
            max_height = self._cfg_float("height", "kneel_max", default=0.70)
            height_up_now = (
                not left_menu_button
                and _face_button_pressed(
                    self.wbcd_config.get_str("buttons", "height_up", default="Y"),
                    a_pressed,
                    b_pressed,
                    x_pressed,
                    y_pressed,
                )
            )
            height_down_now = (
                not left_menu_button
                and _face_button_pressed(
                    self.wbcd_config.get_str("buttons", "height_down", default="X"),
                    a_pressed,
                    b_pressed,
                    x_pressed,
                    y_pressed,
                )
            )
            step = self._cfg_float("height", "kneel_adjust_step_m", default=0.02)
            if height_up_now and not self._prev_wbcd_height_up:
                self.wbcd_height += step
            if height_down_now and not self._prev_wbcd_height_down:
                self.wbcd_height -= step
            self._prev_wbcd_height_up = height_up_now
            self._prev_wbcd_height_down = height_down_now
            self.wbcd_height = clamp(self.wbcd_height, min_height, max_height)
            height = self.wbcd_height
            mode_to_send = LocomotionMode.IDLE_KNEEL_TWO_LEGS
            movement = [0.0, 0.0, 0.0]
            speed = -1.0

        elif self.wbcd_mode == WBCDCompetitionMode.STAND_RECOVERY:
            target = self._cfg_float("height", "stand_recovery_target", default=0.74)
            if self._kneel_recovery_stage != WBCDKneelRecoveryStage.NONE:
                if self.wbcd_recovery_hold:
                    self._kneel_recovery_stage_elapsed += self.dt
                    if (
                        self._kneel_recovery_stage == WBCDKneelRecoveryStage.KNEEL_HOLD
                        and self._kneel_recovery_stage_elapsed
                        >= self._cfg_float("recovery", "kneel_hold_sec", default=0.30)
                    ):
                        self._kneel_recovery_stage = WBCDKneelRecoveryStage.ONE_KNEEL_HOLD
                        self._kneel_recovery_stage_elapsed = 0.0
                        self.wbcd_height = self._cfg_float(
                            "recovery", "one_kneel_height", default=0.50
                        )
                        print(f"[WBCD] Kneel recovery -> {self._kneel_recovery_stage.name}")
                    elif (
                        self._kneel_recovery_stage == WBCDKneelRecoveryStage.ONE_KNEEL_HOLD
                        and self._kneel_recovery_stage_elapsed
                        >= self._cfg_float("recovery", "one_kneel_hold_sec", default=0.60)
                    ):
                        self._kneel_recovery_stage = WBCDKneelRecoveryStage.SQUAT_HOLD
                        self._kneel_recovery_stage_elapsed = 0.0
                        self.wbcd_height = self._cfg_float(
                            "recovery", "squat_recovery_height", default=0.58
                        )
                        print(f"[WBCD] Kneel recovery -> {self._kneel_recovery_stage.name}")
                    elif (
                        self._kneel_recovery_stage == WBCDKneelRecoveryStage.SQUAT_HOLD
                        and self._kneel_recovery_stage_elapsed
                        >= self._cfg_float("recovery", "squat_hold_sec", default=0.50)
                    ):
                        self._kneel_recovery_stage = WBCDKneelRecoveryStage.SQUAT_TO_STAND
                        self._kneel_recovery_stage_elapsed = 0.0
                        print(f"[WBCD] Kneel recovery -> {self._kneel_recovery_stage.name}")
                    elif self._kneel_recovery_stage == WBCDKneelRecoveryStage.SQUAT_TO_STAND:
                        rise_speed = self._cfg_float(
                            "recovery", "squat_to_stand_speed_mps", default=0.06
                        )
                        self.wbcd_height = min(target, self.wbcd_height + rise_speed * self.dt)
                        if self.wbcd_height >= target - 1e-4:
                            self._kneel_recovery_stage = WBCDKneelRecoveryStage.NONE
                            self._kneel_recovery_stage_elapsed = 0.0
                            self._wbcd_return_to_pose = True

                if self._kneel_recovery_stage == WBCDKneelRecoveryStage.KNEEL_HOLD:
                    mode_to_send = LocomotionMode.IDLE_KNEEL_TWO_LEGS
                    height = self.wbcd_height
                elif self._kneel_recovery_stage == WBCDKneelRecoveryStage.ONE_KNEEL_HOLD:
                    mode_to_send = LocomotionMode.IDLE_KNEEL
                    height = self.wbcd_height
                else:
                    mode_to_send = LocomotionMode.IDLE_SQUAT
                    height = self.wbcd_height

                movement = [0.0, 0.0, 0.0]
                speed = -1.0
                if self._cfg_bool("recovery", "allow_recovery_motion", default=False):
                    max_vx = self._cfg_float("movement", "stand_recovery_max_vx", default=0.08)
                    max_wz = self._cfg_float("movement", "stand_recovery_max_wz", default=0.25)
                    movement, facing, speed, mode_to_send_motion = self._compute_wbcd_movement(
                        max_vx, max_wz
                    )
                    if mode_to_send_motion == LocomotionMode.SLOW_WALK:
                        movement[0] = min(0.0, movement[0])
            else:
                if self.wbcd_recovery_hold:
                    rise_speed = self._cfg_float("height", "stand_recovery_speed_mps", default=0.10)
                    self.wbcd_height = min(target, self.wbcd_height + rise_speed * self.dt)
                    if self.wbcd_height >= target - 1e-4:
                        self._wbcd_return_to_pose = True
                height = self.wbcd_height
                max_vx = self._cfg_float("movement", "stand_recovery_max_vx", default=0.08)
                max_wz = self._cfg_float("movement", "stand_recovery_max_wz", default=0.25)
                movement, facing, speed, mode_to_send = self._compute_wbcd_movement(max_vx, max_wz)

        vr_targets = self._collect_vr3pt_targets()
        if vr_targets is None:
            return
        (
            vr_3pt_position,
            vr_3pt_orientation,
            left_hand_position,
            right_hand_position,
        ) = vr_targets
        if self._cfg_bool("entry", "print_vr_target_debug", default=False):
            print(
                "[WBCD] VR targets "
                f"L={vr_3pt_position[0:3]}, R={vr_3pt_position[3:6]}, "
                f"T={vr_3pt_position[6:9]}, "
                f"lean={self.wbcd_lean_pitch_deg:.1f} deg, "
                f"height={height:.3f}, mode={mode_to_send.name}"
            )

        msg = build_planner_message(
            mode_to_send.value,
            movement,
            facing,
            speed=speed,
            height=height,
            upper_body_position=None,
            left_hand_position=left_hand_position,
            right_hand_position=right_hand_position,
            vr_3pt_position=vr_3pt_position,
            vr_3pt_orientation=vr_3pt_orientation,
            vr_3pt_compliance=None,
        )
        self.socket.send(msg)

    def _pace_loop(self):
        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()

    def run_once(self, stream_mode: StreamMode):
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced, in case of headset disconnect
            xrt_timestamp = xrt.get_time_stamp_ns()
            if xrt_timestamp == self.last_xrt_timestamp:
                return
            self.last_xrt_timestamp = xrt_timestamp

            # A+B => next mode; X+Y => previous mode (rising edges)
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            if self.wbcd_enabled() and self.is_wbcd_active():
                self._run_wbcd_once(a_pressed, b_pressed, x_pressed, y_pressed)
                self._pace_loop()
                return

            ab_now = bool(a_pressed) and bool(b_pressed)
            xy_now = bool(x_pressed) and bool(y_pressed)
            if ab_now and not self.prev_ab:
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            if xy_now and not self.prev_xy:
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            self.prev_ab = ab_now
            self.prev_xy = xy_now

            # Read axes/joysticks to control movement, facing, speed and mode
            lx, ly, rx, ry = get_controller_axes()

            # Facing from RIGHT stick: continuous yaw based on rx (right = turn right, left = turn left)
            facing = self.yaw_accumulator.update(rx, self.dt)

            raw_mag = np.hypot(lx, ly)
            raw_mag = np.clip(raw_mag, 0.0, 1.0)
            if np.abs(raw_mag) < JOYSTICK_DEADZONE:
                mag = 0.0
                speed = -1.0
                mode_to_send = LocomotionMode.IDLE
            else:
                mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                if mag > 1.0:
                    mag = 1.0
                mode_to_send = self.mode

                if self.mode == LocomotionMode.SLOW_WALK:
                    speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6
                elif self.mode == LocomotionMode.WALK:
                    speed = -1.0
                elif self.mode == LocomotionMode.RUN:
                    speed = 1.5 + 3 * mag  # 1.5 .. 4.5
                else:
                    speed = mag  # default 0 .. 1.0

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            movement_local = np.array([-lx, ly]) * scale
            perp_x, perp_y = -facing[1], facing[0]
            rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
            movement_global = rotation_facing @ movement_local

            movement = [movement_global[0], movement_global[1], 0.0]

            upper_body_position = None
            left_hand_position = None
            right_hand_position = None
            if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                upper_body_position = self.feedback_reader.upper_body_position_target
                left_hand_position = self.feedback_reader.left_hand_position_target
                right_hand_position = self.feedback_reader.right_hand_position_target

            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    print("[PlannerLoop] Sending VR 3-point pose as target")
                    vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                # Compute hand joints from trigger/grip inputs so operator can
                # control hand open/close while in VR 3PT mode
                (
                    left_menu_button,
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = get_controller_inputs()
                lh_joints, rh_joints = compute_hand_joints_from_inputs(
                    self.left_hand_ik_solver,
                    self.right_hand_ik_solver,
                    left_trigger,
                    left_grip,
                    right_trigger,
                    right_grip,
                )
                left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
            )
            self.socket.send(msg)
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise

        # pacing
        self._pace_loop()


def run_pico_manager(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    wbcd_competition_config: str | None = None,
):
    """
    Manager: creates shared PUB socket and runs pose/planner streamers based on current mode.
    Controller input:
      A+X: Toggle between planner and pose mode
      A+B+X+Y: Toggle policy start/stop
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    # Create shared reader and 3-point pose processor
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )

    pose_streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix="PoseLoop",
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        wbcd_config=WBCDCompetitionConfig(
            wbcd_competition_config
            or os.environ.get("WBCD_PICO_COMPETITION_CONFIG")
            or default_wbcd_config_path()
        ),
    )

    # State machine diagram:
    #
    #   Chain 1 (by_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(by)--> PLANNER_FROZEN_UPPER_BODY <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                                         |
    #                                                                    (by)--> POSE
    #
    #   Chain 2 (ax_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(ax)--> PLANNER <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                        |
    #                                                   (ax)--> POSE
    #
    #   Emergency stop from any mode: A+B+X+Y (start_combo) --> OFF
    #   POSE_PAUSE: left_menu_button held --> POSE_PAUSE, released --> POSE
    #
    print("Manager controls: A+X=toggle mode, A+B+X+Y=start/stop policy")
    current_mode = StreamMode.OFF
    # Track which mode VR_3PT was entered from, so left_axis_click returns to it.
    # Will be either PLANNER or PLANNER_FROZEN_UPPER_BODY.
    vr3pt_parent_mode = StreamMode.PLANNER
    prev_toggle_dc = False
    prev_toggle_da = False
    try:
        prev_ax_pressed = False
        prev_by_pressed = False
        prev_start_combo = False
        prev_left_axis_click = False
        prev_wbcd_stand = False
        prev_wbcd_half_squat = False
        prev_wbcd_kneel = False
        while True:
            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()

            left_menu_button, _, _, left_grip_mgr, _ = get_controller_inputs()
            if planner_streamer.wbcd_config is not None:
                planner_streamer.wbcd_config.reload_if_needed()

            left_axis_click, _ = get_axis_clicks()

            wbcd_enabled = planner_streamer.wbcd_enabled()
            if not wbcd_enabled and planner_streamer.is_wbcd_active():
                planner_streamer.clear_wbcd_mode()

            wbcd_stand_pressed = wbcd_enabled and left_menu_button and _face_button_pressed(
                planner_streamer.wbcd_config.get_str("buttons", "stand_manip", default="A"),
                a_pressed,
                b_pressed,
                x_pressed,
                y_pressed,
            )
            wbcd_half_squat_pressed = wbcd_enabled and left_menu_button and _face_button_pressed(
                planner_streamer.wbcd_config.get_str("buttons", "half_squat_manip", default="X"),
                a_pressed,
                b_pressed,
                x_pressed,
                y_pressed,
            )
            wbcd_kneel_pressed = wbcd_enabled and left_menu_button and _face_button_pressed(
                planner_streamer.wbcd_config.get_str("buttons", "kneel_manip", default="Y"),
                a_pressed,
                b_pressed,
                x_pressed,
                y_pressed,
            )
            wbcd_recovery_pressed = wbcd_enabled and left_menu_button and _face_button_pressed(
                planner_streamer.wbcd_config.get_str("buttons", "hold_stand_recovery", default="B"),
                a_pressed,
                b_pressed,
                x_pressed,
                y_pressed,
            )

            # Existing Sonic combos are suppressed while left_menu is held for WBCD chords.
            ax_pressed = (a_pressed) and (x_pressed) and not left_menu_button
            by_pressed = (b_pressed) and (y_pressed) and not left_menu_button

            # Rising edge: A+B+X+Y pressed together -> toggle policy start/stop.
            start_combo = (a_pressed) and (b_pressed) and (x_pressed) and (y_pressed)

            if wbcd_enabled and planner_streamer.is_wbcd_active():
                current_wbcd_mode = planner_streamer.wbcd_mode
                if wbcd_stand_pressed and not prev_wbcd_stand:
                    planner_streamer.enter_wbcd_mode(WBCDCompetitionMode.STAND_MANIP)
                elif wbcd_half_squat_pressed and not prev_wbcd_half_squat:
                    planner_streamer.enter_wbcd_mode(WBCDCompetitionMode.HALF_SQUAT_MANIP)
                elif wbcd_kneel_pressed and not prev_wbcd_kneel:
                    if current_wbcd_mode in (
                        WBCDCompetitionMode.STAND_MANIP,
                        WBCDCompetitionMode.HALF_SQUAT_MANIP,
                        WBCDCompetitionMode.KNEEL_MANIP,
                    ):
                        planner_streamer.enter_wbcd_mode(WBCDCompetitionMode.KNEEL_MANIP)
                    else:
                        print("[WBCD] Enter HALF_SQUAT_MANIP before KNEEL_MANIP")
                elif current_wbcd_mode != WBCDCompetitionMode.STAND_RECOVERY:
                    if wbcd_recovery_pressed:
                        if current_wbcd_mode == WBCDCompetitionMode.KNEEL_MANIP:
                            planner_streamer.start_kneel_recovery()
                        else:
                            planner_streamer.enter_wbcd_mode(WBCDCompetitionMode.STAND_RECOVERY)
                            planner_streamer.set_wbcd_recovery_hold(True)
                else:
                    planner_streamer.set_wbcd_recovery_hold(wbcd_recovery_pressed)

                if start_combo and not prev_start_combo:
                    planner_streamer.clear_wbcd_mode()
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()
                if ax_pressed and not prev_ax_pressed:
                    planner_streamer.clear_wbcd_mode()
                    pose_streamer.reset_yaw()
                    socket.send(build_command_message(start=True, stop=False, planner=False))
                    print("[Manager] StreamMode switch: PLANNER_VR_3PT -> POSE")
                    current_mode = StreamMode.POSE
                    prev_ax_pressed = ax_pressed
                    prev_by_pressed = by_pressed
                    prev_start_combo = start_combo
                    prev_left_axis_click = left_axis_click
                    prev_wbcd_stand = wbcd_stand_pressed
                    prev_wbcd_half_squat = wbcd_half_squat_pressed
                    prev_wbcd_kneel = wbcd_kneel_pressed
                    continue
                if by_pressed and not prev_by_pressed:
                    planner_streamer.clear_wbcd_mode()
                    pose_streamer.reset_yaw()
                    socket.send(build_command_message(start=True, stop=False, planner=False))
                    print("[Manager] StreamMode switch: PLANNER_VR_3PT -> POSE")
                    current_mode = StreamMode.POSE
                    prev_ax_pressed = ax_pressed
                    prev_by_pressed = by_pressed
                    prev_start_combo = start_combo
                    prev_left_axis_click = left_axis_click
                    prev_wbcd_stand = wbcd_stand_pressed
                    prev_wbcd_half_squat = wbcd_half_squat_pressed
                    prev_wbcd_kneel = wbcd_kneel_pressed
                    continue
                else:
                    new_mode = StreamMode.PLANNER_VR_3PT
                    planner_streamer.run_once(new_mode)
                    if planner_streamer.consume_wbcd_return_to_pose():
                        planner_streamer.clear_wbcd_mode()
                        socket.send(build_command_message(start=True, stop=False, planner=False))
                        print("[Manager] StreamMode switch: PLANNER_VR_3PT -> POSE")
                        current_mode = StreamMode.POSE
                        prev_ax_pressed = ax_pressed
                        prev_by_pressed = by_pressed
                        prev_start_combo = start_combo
                        prev_left_axis_click = left_axis_click
                        prev_wbcd_stand = wbcd_stand_pressed
                        prev_wbcd_half_squat = wbcd_half_squat_pressed
                        prev_wbcd_kneel = wbcd_kneel_pressed
                        continue

                    socket.send(
                        pack_pose_message(
                            {
                                "stream_mode": np.array([new_mode.value], dtype=np.int32),
                                "toggle_data_collection": np.array([False], dtype=bool),
                                "toggle_data_abort": np.array([False], dtype=bool),
                            },
                            topic="manager_state",
                        )
                    )
                    prev_ax_pressed = ax_pressed
                    prev_by_pressed = by_pressed
                    prev_start_combo = start_combo
                    prev_left_axis_click = left_axis_click
                    prev_wbcd_stand = wbcd_stand_pressed
                    prev_wbcd_half_squat = wbcd_half_squat_pressed
                    prev_wbcd_kneel = wbcd_kneel_pressed
                    continue

            new_mode = current_mode
            if current_mode == StreamMode.OFF:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.PLANNER
                    # Calibrate VR 3pt tracking NOW: operator should be in zero-ref pose.
                    # Uses the current Pico SMPL frame + FK of all-zero body joints.
                    sample = reader.get_latest()
                    if sample is not None:
                        three_point.calibrate_now(sample["body_poses_np"])
                    else:
                        print("[Manager] WARNING: No SMPL data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                # Chain 2: POSE <--(ax)--> PLANNER <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif wbcd_stand_pressed and not prev_wbcd_stand:
                    if planner_streamer.prepare_wbcd_vr3pt_entry(WBCDCompetitionMode.STAND_MANIP):
                        new_mode = StreamMode.PLANNER_VR_3PT
                        planner_streamer.enter_wbcd_mode(WBCDCompetitionMode.STAND_MANIP)
                elif wbcd_half_squat_pressed and not prev_wbcd_half_squat:
                    if planner_streamer.prepare_wbcd_vr3pt_entry(
                        WBCDCompetitionMode.HALF_SQUAT_MANIP
                    ):
                        new_mode = StreamMode.PLANNER_VR_3PT
                        planner_streamer.enter_wbcd_mode(WBCDCompetitionMode.HALF_SQUAT_MANIP)
                elif wbcd_kneel_pressed and not prev_wbcd_kneel:
                    print("[WBCD] Enter HALF_SQUAT_MANIP before KNEEL_MANIP")
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER  # Enter chain 2
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY  # Enter chain 1
                elif left_menu_button:
                    new_mode = StreamMode.POSE_PAUSE

            elif current_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                # Chain 1: POSE <--(by)--> FROZEN <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE_PAUSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif not left_menu_button:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                # VR_3PT is reachable from both chains:
                #   left_axis_click → return to parent (PLANNER or FROZEN)
                #   ax_pressed      → POSE (chain 2 exit)
                #   by_pressed      → POSE (chain 1 exit)
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = vr3pt_parent_mode  # Return to parent mode
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.POSE

            # Handle mode transitions before running loop
            if new_mode != current_mode:
                if current_mode == StreamMode.POSE:
                    pose_streamer.on_mode_exit()
                if new_mode != StreamMode.PLANNER_VR_3PT:
                    planner_streamer.clear_wbcd_mode()

                # Track parent when entering VR_3PT
                if new_mode == StreamMode.PLANNER_VR_3PT:
                    vr3pt_parent_mode = current_mode
                    print(f"[Manager] VR_3PT parent: {vr3pt_parent_mode.name}")

                if new_mode == StreamMode.POSE:
                    pose_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER and current_mode != StreamMode.PLANNER_VR_3PT:
                    # Only reset yaw when freshly entering PLANNER from POSE,
                    # not when returning from VR_3PT sub-mode
                    planner_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                    if current_mode != StreamMode.PLANNER_VR_3PT:
                        # Freshly entering from POSE: reset yaw and grab initial targets
                        planner_streamer.reset_yaw()
                    # Always re-grab the latest robot state as frozen targets,
                    # whether entering from POSE or returning from VR_3PT
                    # (the old targets are stale after VR_3PT moved the arms)
                    planner_streamer.save_upper_body_position_target()
                elif (
                    new_mode == StreamMode.PLANNER_VR_3PT
                    and not planner_streamer.is_wbcd_active()
                ):
                    # Recalibrate VR tracking against the robot's actual current pose
                    # (read via g1_debug feedback + FK) to prevent sudden jumps
                    planner_streamer.recalibrate_for_vr3pt()

            # Run one iteration of the new mode
            if new_mode == StreamMode.POSE:
                pose_streamer.run_once()
            elif (
                new_mode == StreamMode.PLANNER
                or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                or new_mode == StreamMode.PLANNER_VR_3PT
            ):
                planner_streamer.run_once(new_mode)

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()
                elif (
                    new_mode == StreamMode.PLANNER
                    or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                    or new_mode == StreamMode.PLANNER_VR_3PT
                ):
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                elif new_mode == StreamMode.POSE:
                    socket.send(build_command_message(start=True, stop=False, planner=False))

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            # Mode-independent: send manager_state for data exporter
            toggle_dc_tmp = bool(a_pressed) and left_grip_mgr > 0.5
            toggle_da_tmp = bool(b_pressed) and left_grip_mgr > 0.5
            toggle_dc = toggle_dc_tmp and not prev_toggle_dc
            toggle_da = toggle_da_tmp and not prev_toggle_da
            prev_toggle_dc = toggle_dc_tmp
            prev_toggle_da = toggle_da_tmp
            socket.send(
                pack_pose_message(
                    {
                        "stream_mode": np.array([current_mode.value], dtype=np.int32),
                        "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                        "toggle_data_abort": np.array([toggle_da], dtype=bool),
                    },
                    topic="manager_state",
                )
            )

            prev_ax_pressed = ax_pressed
            prev_by_pressed = by_pressed
            prev_start_combo = start_combo
            prev_left_axis_click = left_axis_click
            prev_wbcd_stand = wbcd_stand_pressed
            prev_wbcd_half_squat = wbcd_half_squat_pressed
            prev_wbcd_kneel = wbcd_kneel_pressed

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 200)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    parser.add_argument(
        "--wbcd_competition_config",
        type=str,
        default=None,
        help="Path to WBCD competition teleop YAML config",
    )
    args = parser.parse_args()

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            wbcd_competition_config=args.wbcd_competition_config,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
