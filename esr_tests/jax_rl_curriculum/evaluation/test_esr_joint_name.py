"""
Test the trajectory structure of the ESR environment without loading
or running a trained RL agent.

The script checks:
    - trajectory joint names
    - whether ankle_angle_r is contained in the trajectory
    - trajectory joint_name2ind_qpos mapping
    - MuJoCo qpos addresses
    - whether esr_hinge_r is part of the trajectory
    - the ESR hinge trajectory value at the first trajectory sample

Run this script from the LocoMuJoCo project environment.
"""

import os
import argparse
import numpy as np
import mujoco

from loco_mujoco import TaskFactory
from omegaconf import OmegaConf


# =============================================================================
# Environment setup
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the training config YAML file."
)
args = parser.parse_args()

# Headless MuJoCo
os.environ["MUJOCO_GL"] = "egl"

print("=" * 70)
print("Creating ESR environment WITHOUT RL agent")
print("=" * 70)

# -------------------------------------------------------------------------
# Load config
# -------------------------------------------------------------------------
config = OmegaConf.load(args.config)

OmegaConf.set_struct(config, False)

config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True

# -------------------------------------------------------------------------
# Task factory
# -------------------------------------------------------------------------
factory = TaskFactory.get_factory_cls(
    config.experiment.task_factory.name
)

domain_randomization_type = config.randomization_config["randomization_type"]

domain_randomization_params = {
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {},

    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {},

    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {
        "pylon_socket": {
            "x": [-0.0, 0.0],
            "z": [-0.0, 0.0],
        },
        "talus": {
            "x": [-0.0, 0.0],
            "z": [-0.0, 0.0],
        },
    },

    "randomize_prosthesis_body_orientation": True,
    "prosthesis_body_orientation_range": {
        "pylon_socket": {
            "x": [-0.0, 0.0],
            "y": [-0.0, 0.0],
            "z": [-0.0, 0.0],
        },
        "talus": {
            "x": [-0.0, 0.0],
            "z": [-0.0, 0.0],
        },
    },
}

print("\nCreating environment...")

env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    domain_randomization_type=domain_randomization_type,
    domain_randomization_params=domain_randomization_params,
)

print("Environment created.")


# =============================================================================
# Reset environment
# =============================================================================

print("\nResetting environment...")

# Depending on the environment implementation, reset may need a key.
# For this diagnostic, use the environment's own reset if possible.
try:
    env_state = env.reset()
except TypeError:
    import jax
    rng = jax.random.key(0)
    obs, env_state = env.reset(rng)

print("Environment reset successful.")


# =============================================================================
# 1) TRAJECTORY INFO
# =============================================================================

print("\n" + "=" * 70)
print("TRAJECTORY INFO")
print("=" * 70)

traj_info = env.th.traj.info

print("\njoint_names:")
print(traj_info.joint_names)

print("\nContains ankle_angle_r?")
print("ankle_angle_r" in traj_info.joint_names)

print("\njoint_name2ind_qpos:")
print(traj_info.joint_name2ind_qpos)


# =============================================================================
# 2) Trajectory qpos dimensions
# =============================================================================

print("\n" + "=" * 70)
print("TRAJECTORY vs. MUJOCO qPOS")
print("=" * 70)

traj_sample = env.th.traj.data.get(0, 0, np)

print("\nTrajectory qpos shape:")
print(traj_sample.qpos.shape)

print("\nTrajectory qpos dimension:")
print(len(traj_sample.qpos))

print("\nMuJoCo model nq:")
print(env._model.nq)

print("\nMuJoCo model nv:")
print(env._model.nv)


# =============================================================================
# 3) Print trajectory joint indices
# =============================================================================

print("\n" + "=" * 70)
print("TRAJECTORY JOINT INDICES")
print("=" * 70)

for joint_name, idx in traj_info.joint_name2ind_qpos.items():
    print(f"{joint_name:30s} -> trajectory qpos index {idx}")


# =============================================================================
# 4) ESR hinge in MuJoCo model
# =============================================================================

print("\n" + "=" * 70)
print("ESR HINGE IN MUJOCO MODEL")
print("=" * 70)

mj_model = env._model

hinge_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "esr_hinge_r",
)

if hinge_id < 0:
    print("ERROR: esr_hinge_r was NOT found in the MuJoCo model.")
else:
    hinge_qpos = int(mj_model.jnt_qposadr[hinge_id])
    hinge_qvel = int(mj_model.jnt_dofadr[hinge_id])

    print("ESR hinge joint ID:")
    print(hinge_id)

    print("\nESR hinge MuJoCo qpos address:")
    print(hinge_qpos)

    print("\nESR hinge MuJoCo qvel address:")
    print(hinge_qvel)


# =============================================================================
# 5) Is ESR hinge part of the trajectory?
# =============================================================================

print("\n" + "=" * 70)
print("ESR HINGE IN TRAJECTORY")
print("=" * 70)

hinge_in_traj = "esr_hinge_r" in traj_info.joint_names

print("\nContains esr_hinge_r?")
print(hinge_in_traj)

if "esr_hinge_r" in traj_info.joint_name2ind_qpos:
    traj_hinge_qpos = traj_info.joint_name2ind_qpos["esr_hinge_r"]

    print("\nTrajectory qpos index of esr_hinge_r:")
    print(traj_hinge_qpos)

    print("\nTrajectory value at sample (0, 0):")
    print(traj_sample.qpos[traj_hinge_qpos])

else:
    print("\nNo trajectory index for esr_hinge_r found.")


# =============================================================================
# 6) IMPORTANT: original test using MuJoCo address
# =============================================================================

print("\n" + "=" * 70)
print("CHECK USING MUJOCO qPOS ADDRESS")
print("=" * 70)

if hinge_id >= 0:

    hinge_qpos = int(mj_model.jnt_qposadr[hinge_id])

    print("\nMuJoCo esr_hinge_r qpos address:")
    print(hinge_qpos)

    print("\nTrying traj_sample.qpos[hinge_qpos]:")

    if hinge_qpos < len(traj_sample.qpos):
        print(traj_sample.qpos[hinge_qpos])
        print(
            "\nWARNING: This value is only meaningful if the trajectory "
            "qpos uses the same indexing as the full MuJoCo model."
        )
    else:
        print(
            "OUT OF RANGE! The trajectory qpos has only "
            f"{len(traj_sample.qpos)} entries."
        )


# =============================================================================
# 7) Compare ankle_angle_r and ESR hinge
# =============================================================================

print("\n" + "=" * 70)
print("ANKLE / ESR HINGE COMPARISON")
print("=" * 70)

if "ankle_angle_r" in traj_info.joint_name2ind_qpos:

    ankle_idx = traj_info.joint_name2ind_qpos["ankle_angle_r"]

    print("\nTrajectory index of ankle_angle_r:")
    print(ankle_idx)

    print("\nankle_angle_r trajectory value:")
    print(traj_sample.qpos[ankle_idx])

else:
    print("\nankle_angle_r is NOT part of the trajectory.")

if "esr_hinge_r" in traj_info.joint_name2ind_qpos:

    hinge_idx = traj_info.joint_name2ind_qpos["esr_hinge_r"]

    print("\nesr_hinge_r trajectory index:")
    print(hinge_idx)

    print("\nesr_hinge_r trajectory value:")
    print(traj_sample.qpos[hinge_idx])

else:
    print("\nesr_hinge_r is NOT part of the trajectory.")


# =============================================================================
# 8) Final summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Trajectory qpos dimension : {len(traj_sample.qpos)}")
print(f"MuJoCo nq                 : {mj_model.nq}")
print(f"ankle_angle_r in traj     : {'ankle_angle_r' in traj_info.joint_names}")
print(f"esr_hinge_r in traj       : {'esr_hinge_r' in traj_info.joint_names}")

if hinge_id >= 0:
    print(f"esr_hinge_r MuJoCo qpos   : {int(mj_model.jnt_qposadr[hinge_id])}")

if "esr_hinge_r" in traj_info.joint_name2ind_qpos:
    print(
        "esr_hinge_r traj qpos     : "
        f"{traj_info.joint_name2ind_qpos['esr_hinge_r']}"
    )

print("=" * 70)