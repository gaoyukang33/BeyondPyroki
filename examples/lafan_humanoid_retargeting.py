"""LaFAN Batch Humanoid Retargeting

Retarget all LaFAN .npy files in an input directory to the G1 humanoid robot,
and save results to an output directory with corresponding filenames.

Based on example 12 (fancy retargeting) for better quality:
includes scale regularization, root smoothness, and self-collision avoidance.

Usage:
    python lafan_humanoid_retargeting.py --input_dir /path/to/lafan --output_dir /path/to/output
"""

import argparse
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

from retarget_helpers._utils import (
    LAFAN_JOINT_NAMES,
    create_conn_tree,
    get_lafan_retarget_indices,
)


def transform_y_up_to_z_up(joints: onp.ndarray) -> onp.ndarray:
    """Convert Y-up coordinates to Z-up: [x, y, z] -> [x, z, y]."""
    transform_matrix = onp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return joints @ transform_matrix.T


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    root_smoothness: float


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_keypoints: jnp.ndarray,
    source_joint_retarget_indices: jnp.ndarray,
    g1_joint_retarget_indices: jnp.ndarray,
    source_mask: jnp.ndarray,
    weights: RetargetingWeights,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the retargeting problem for LaFAN data."""

    n_retarget = len(source_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    joints_to_move_less = jnp.array(
        [
            robot.joints.actuated_names.index(name)
            for name in ["left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint"]
        ]
    )

    # Variables.
    class SourceJointsScaleVar(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_source_joints_scale = SourceJointsScaleVar(jnp.zeros(timesteps))
    var_offset = OffsetVar(jnp.zeros(timesteps))

    # ---- Cost definitions ----

    @jaxls.Cost.factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_source_joints_scale: SourceJointsScaleVar,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Match relative joint positions and angles between source and robot."""
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        source_pos = keypoints[jnp.array(source_joint_retarget_indices)]
        robot_pos = T_world_link.translation()[jnp.array(g1_joint_retarget_indices)]

        delta_source = source_pos[:, None] - source_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        position_scale = var_values[var_source_joints_scale][..., None]
        residual_position_delta = (
            (delta_source - delta_robot * position_scale)
            * (1 - jnp.eye(delta_source.shape[0])[..., None])
            * source_mask[..., None]
        )

        delta_source_normalized = delta_source / jnp.linalg.norm(
            delta_source + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (delta_source_normalized * delta_robot_normalized).sum(
            axis=-1
        )
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * source_mask
        )

        return (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )

    @jaxls.Cost.factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_source_joints_scale: SourceJointsScaleVar,
    ) -> jax.Array:
        """Regularize scale: close to 1, symmetric, non-negative."""
        scale = var_values[var_source_joints_scale]
        res_close_to_one = (scale - 1.0).flatten() * 1.0
        res_symmetric = (scale - scale.T).flatten() * 100.0
        res_nonneg = jnp.clip(-scale, min=0).flatten() * 100.0
        return jnp.concatenate([res_close_to_one, res_symmetric, res_nonneg])

    @jaxls.Cost.factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Align keypoint positions to robot link positions in world frame."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[g1_joint_retarget_indices]
        keypoint_pos = keypoints[source_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    @jaxls.Cost.factory
    def root_smoothness_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Penalize jittery root pose changes."""
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    # ---- Assemble costs ----

    costs = [
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_source_joints_scale,
            target_keypoints,
        ),
        scale_regularization(var_source_joints_scale),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.2]),
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(var_joints.default_factory().shape, 0.2)
            .at[joints_to_move_less]
            .set(2.0)[None],
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.05,
            weight=2.0,
        ),
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_keypoints,
        ),
        root_smoothness_cost(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
        ),
    ]

    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[
                var_joints,
                var_Ts_world_root,
                var_source_joints_scale,
                var_offset,
            ],
        )
        .analyze()
        .solve()
    )
    transform = solution[var_Ts_world_root]
    offset = solution[var_offset]
    transform = jaxlie.SE3.from_translation(offset) @ transform
    return transform, solution[var_joints]


def retarget_single_file(
    npy_path: Path,
    output_path: Path,
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    lafan_joint_retarget_indices: jnp.ndarray,
    g1_joint_retarget_indices: jnp.ndarray,
    lafan_mask: jnp.ndarray,
    weights: RetargetingWeights,
    max_frames: int,
    subsample: int,
):
    """Retarget a single LaFAN .npy file and save results."""
    lafan_keypoints = onp.load(str(npy_path))
    assert lafan_keypoints.shape[1:] == (22, 3), (
        f"{npy_path.name}: expected (N, 22, 3), got {lafan_keypoints.shape}"
    )

    # Coordinate transform and subsampling.
    lafan_keypoints = transform_y_up_to_z_up(lafan_keypoints)
    if subsample > 1:
        lafan_keypoints = lafan_keypoints[::subsample]
    if max_frames > 0:
        lafan_keypoints = lafan_keypoints[:max_frames]

    num_timesteps = lafan_keypoints.shape[0]
    print(f"  Solving {num_timesteps} frames ...")

    Ts_world_root, joints = solve_retargeting(
        robot=robot,
        robot_coll=robot_coll,
        target_keypoints=lafan_keypoints,
        source_joint_retarget_indices=lafan_joint_retarget_indices,
        g1_joint_retarget_indices=g1_joint_retarget_indices,
        source_mask=lafan_mask,
        weights=weights,
    )

    result = {
        "joints": onp.array(joints),                      # (N, n_dof)
        "transforms": onp.array(Ts_world_root.wxyz_xyz),  # (N, 7) wxyz_xyz
    }
    onp.savez(str(output_path), **result)
    print(f"  Saved -> {output_path}  (joints: {result['joints'].shape}, transforms: {result['transforms'].shape})")


def main():
    parser = argparse.ArgumentParser(description="Batch LaFAN -> G1 retargeting")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/yukanggao/Desktop/code_for_nips/holosoma/src/holosoma_retargeting"
                "/holosoma_retargeting/demo_data/lafan",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/yukanggao/Desktop/code_for_nips/pyroki/examples/retarget_output",
    )
    parser.add_argument("--max_frames", type=int, default=500, help="Max frames per file (0 = all)")
    parser.add_argument("--subsample", type=int, default=4, help="Take every N-th frame (default 4: 30fps->7.5fps)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subsample: {args.subsample}, Max frames: {args.max_frames}")

    # Load robot once.
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    lafan_joint_retarget_indices, g1_joint_retarget_indices = get_lafan_retarget_indices()
    lafan_mask = create_conn_tree(robot, g1_joint_retarget_indices)

    weights: RetargetingWeights = {
        "local_alignment": 2.0,
        "global_alignment": 1.0,
        "root_smoothness": 1.0,
    }

    for i, npy_path in enumerate(npy_files):
        print(f"\n[{i + 1}/{len(npy_files)}] Processing: {npy_path.name}")
        output_path = output_dir / f"{npy_path.stem}.npz"
        retarget_single_file(
            npy_path=npy_path,
            output_path=output_path,
            robot=robot,
            robot_coll=robot_coll,
            lafan_joint_retarget_indices=lafan_joint_retarget_indices,
            g1_joint_retarget_indices=g1_joint_retarget_indices,
            lafan_mask=lafan_mask,
            weights=weights,
            max_frames=args.max_frames,
            subsample=args.subsample,
        )

    print(f"\nDone! All results saved to {output_dir}")


if __name__ == "__main__":
    main()
