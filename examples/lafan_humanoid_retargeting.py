"""LaFAN Humanoid Retargeting

Retarget LaFAN motion capture data (22 joints, Y-up) to the G1 humanoid robot.
Based on 10_humanoid_retargeting.py, adapted for the LaFAN joint format.
"""

import time
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from retarget_helpers._utils import (
    LAFAN_JOINT_NAMES,
    create_conn_tree,
    get_lafan_retarget_indices,
)


# LaFAN data uses Y-up; pyroki uses Z-up.
def transform_y_up_to_z_up(joints: onp.ndarray) -> onp.ndarray:
    """Convert Y-up coordinates to Z-up: [x, y, z] -> [x, z, -y]."""
    result = onp.empty_like(joints)
    result[..., 0] = joints[..., 0]
    result[..., 1] = joints[..., 2]
    result[..., 2] = -joints[..., 1]
    return result


class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""


def main():
    """Main function for LaFAN humanoid retargeting."""

    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Load LaFAN motion data: [N, 22, 3] in Y-up coordinate system.
    lafan_path = Path(
        "/Users/yukanggao/Desktop/code_for_nips/holosoma/src/holosoma_retargeting"
        "/holosoma_retargeting/demo_data/lafan/aiming1_subject1.npy"
    )
    lafan_keypoints = onp.load(str(lafan_path))  # (7184, 22, 3)
    assert lafan_keypoints.shape[1:] == (22, 3), (
        f"Expected (N, 22, 3), got {lafan_keypoints.shape}"
    )

    # Convert Y-up to Z-up.
    lafan_keypoints = transform_y_up_to_z_up(lafan_keypoints)

    # Subsample to make optimization tractable.
    # Use every 4th frame (~7.5 FPS from 30 FPS), and limit to first 500 frames.
    max_frames = 500
    subsample = 4
    lafan_keypoints = lafan_keypoints[::subsample][:max_frames]

    num_timesteps = lafan_keypoints.shape[0]
    num_joints = lafan_keypoints.shape[1]
    print(f"Retargeting {num_timesteps} frames, {num_joints} joints")

    lafan_joint_retarget_indices, g1_joint_retarget_indices = (
        get_lafan_retarget_indices()
    )
    lafan_mask = create_conn_tree(robot, g1_joint_retarget_indices)

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)

    weights = pk.viewer.WeightTuner(
        server,
        RetargetingWeights(  # type: ignore
            local_alignment=2.0,
            global_alignment=1.0,
        ),
    )

    Ts_world_root, joints = None, None

    def generate_trajectory():
        nonlocal Ts_world_root, joints
        gen_button.disabled = True
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=lafan_keypoints,
            source_joint_retarget_indices=lafan_joint_retarget_indices,
            g1_joint_retarget_indices=g1_joint_retarget_indices,
            source_mask=lafan_mask,
            weights=weights.get_weights(),  # type: ignore
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
            urdf_vis.update_cfg(onp.array(joints[tstep]))
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(lafan_keypoints[tstep]),
                onp.array((0, 0, 255))[None].repeat(num_joints, axis=0),
                point_size=0.01,
            )

        time.sleep(0.05)


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    source_joint_retarget_indices: jnp.ndarray,
    g1_joint_retarget_indices: jnp.ndarray,
    source_mask: jnp.ndarray,
    weights: RetargetingWeights,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the retargeting problem for LaFAN data."""

    n_retarget = len(source_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Joints that should move less for natural humanoid motion.
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

    # Costs and constraints.
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

        # NxN grid of relative positions.
        delta_source = source_pos[:, None] - source_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        position_scale = var_values[var_source_joints_scale][..., None]
        residual_position_delta = (
            (delta_source - delta_robot * position_scale)
            * (1 - jnp.eye(delta_source.shape[0])[..., None])
            * source_mask[..., None]
        )

        # Vector angle regularization.
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

        residual = (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )
        return residual

    @jaxls.Cost.factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Soft cost to align the human keypoints to the robot, in the world frame."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[g1_joint_retarget_indices]
        keypoint_pos = keypoints[source_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    costs = [
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_source_joints_scale,
            target_keypoints,
        ),
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
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_keypoints,
        ),
    ]

    costs.append(
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
        ),
    )

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


if __name__ == "__main__":
    main()
