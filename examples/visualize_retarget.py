"""Visualize Retargeting Results

Side-by-side playback of the original LaFAN reference motion (point cloud + skeleton)
and the retargeted G1 robot motion in Viser.

Usage:
    python visualize_retarget.py \
        --lafan /path/to/lafan/aiming1_subject1.npy \
        --retarget /path/to/retarget_output/aiming1_subject1.npz
"""

import argparse
import time

import numpy as onp
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from retarget_helpers._utils import LAFAN_JOINT_NAMES

# LaFAN skeleton connectivity for drawing bones.
LAFAN_BONE_PAIRS = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("Spine2", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToeBase"),
]

BONE_INDEX_PAIRS = [
    (LAFAN_JOINT_NAMES.index(a), LAFAN_JOINT_NAMES.index(b))
    for a, b in LAFAN_BONE_PAIRS
]


def transform_y_up_to_z_up(joints: onp.ndarray) -> onp.ndarray:
    """Convert Y-up coordinates to Z-up: [x, y, z] -> [x, z, y]."""
    transform_matrix = onp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return joints @ transform_matrix.T


def main():
    parser = argparse.ArgumentParser(description="Visualize retargeting: reference + G1")
    parser.add_argument(
        "--lafan",
        type=str,
        required=True,
        help="Path to the original LaFAN .npy file (N, 22, 3)",
    )
    parser.add_argument(
        "--retarget",
        type=str,
        required=True,
        help="Path to the retargeted .npz file (joints + transforms)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Must match the subsample used during retargeting",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Must match the max_frames used during retargeting",
    )
    parser.add_argument(
        "--offset_x",
        type=float,
        default=2.0,
        help="Horizontal offset between reference and robot (meters)",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Playback FPS")
    args = parser.parse_args()

    # Load reference LaFAN data and apply same preprocessing as retargeting.
    lafan_keypoints = onp.load(args.lafan)
    lafan_keypoints = transform_y_up_to_z_up(lafan_keypoints)
    if args.subsample > 1:
        lafan_keypoints = lafan_keypoints[:: args.subsample]
    if args.max_frames > 0:
        lafan_keypoints = lafan_keypoints[: args.max_frames]

    # Load retargeted results.
    retarget_data = onp.load(args.retarget)
    joints = retarget_data["joints"]          # (N, n_dof)
    transforms = retarget_data["transforms"]  # (N, 7) wxyz_xyz

    num_timesteps = min(len(lafan_keypoints), len(joints))
    lafan_keypoints = lafan_keypoints[:num_timesteps]
    joints = joints[:num_timesteps]
    transforms = transforms[:num_timesteps]

    print(f"Frames: {num_timesteps}")
    print(f"Reference: {lafan_keypoints.shape}, Robot joints: {joints.shape}")

    # Load G1 robot for visualization.
    urdf = load_robot_description("g1_description")

    # Start Viser server.
    server = viser.ViserServer()

    # --- Robot (retargeted) on the right ---
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # --- GUI controls ---
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider(
        "timestep", 0, num_timesteps - 1, 1, 0
    )
    show_skeleton = server.gui.add_checkbox("show skeleton", True)
    show_points = server.gui.add_checkbox("show points", True)

    # Offset: place reference motion to the left of the robot.
    ref_offset = onp.array([-args.offset_x, 0.0, 0.0])

    print(f"Viser running at http://localhost:8080")
    print(f"  Left: reference LaFAN skeleton (blue)")
    print(f"  Right: retargeted G1 robot")

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            t = timestep_slider.value

            # Update robot pose.
            robot_frame.wxyz = transforms[t, :4]
            robot_frame.position = transforms[t, 4:]
            urdf_vis.update_cfg(joints[t])

            # Reference point cloud.
            ref_pts = lafan_keypoints[t] + ref_offset
            if show_points.value:
                colors = onp.full((22, 3), [70, 130, 255], dtype=onp.uint8)
                server.scene.add_point_cloud(
                    "/ref/points",
                    ref_pts.astype(onp.float32),
                    colors,
                    point_size=0.02,
                )
            else:
                server.scene.remove("/ref/points")

            # Reference skeleton lines.
            if show_skeleton.value:
                for k, (i, j) in enumerate(BONE_INDEX_PAIRS):
                    start = ref_pts[i].astype(onp.float32)
                    end = ref_pts[j].astype(onp.float32)
                    mid = (start + end) / 2.0
                    diff = end - start
                    length = float(onp.linalg.norm(diff))
                    if length < 1e-6:
                        continue
                    # Draw each bone as a thin spline/line via add_spline_catmull_rom.
                    server.scene.add_spline_catmull_rom(
                        f"/ref/bone/{k}",
                        positions=onp.stack([start, end]),
                        color=(70, 130, 255),
                        line_width=3.0,
                    )
            else:
                for k in range(len(BONE_INDEX_PAIRS)):
                    server.scene.remove(f"/ref/bone/{k}")

        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
