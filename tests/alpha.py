import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from loco_mujoco import ImitationFactory


# ============================================================
# ENV
# ============================================================

env_params = {
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial",
    "prosthesis_subtype": "SACH",
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2,
    "reattach_muscles": {"med_gas": [0, 0, 0]},
    "remove_joint_names": ["subtalar_angle"],
    "joint_stiffness": {"ankle_angle": 900},
    "adapt_joint_range": {"ankle_angle_r": [-10, 10]},
}

env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    **env_params,
    default_dataset_conf=dict(task="walk"),
)

# ============================================================
# RESET
# ============================================================

key = jax.random.key(0)
state = env.mjx_reset(key)

data = state.data

# ============================================================
# IDs
# ============================================================

distal_id = mujoco.mj_name2id(
    env._model,
    mujoco.mjtObj.mjOBJ_SITE,
    "talus_attachment_site_in_pylon_r",
)

proximal_id = mujoco.mj_name2id(
    env._model,
    mujoco.mjtObj.mjOBJ_SITE,
    "pylon_mimic_r",
)

tx_id = mujoco.mj_name2id(
    env._model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "socket_tx_r",
)

ty_id = mujoco.mj_name2id(
    env._model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "socket_ty_r",
)

tx_qpos = env._model.jnt_qposadr[tx_id]
ty_qpos = env._model.jnt_qposadr[ty_id]

# ============================================================
# BASELINE
# ============================================================

baseline_distal = np.array(data.site_xpos[distal_id])
baseline_prox = np.array(data.site_xpos[proximal_id])

print("\n==============================")
print("BASELINE")
print("==============================")
print("distal site :", baseline_distal)
print("proximal site:", baseline_prox)

# ============================================================
# PYLON AXIS
# ============================================================

pylon_axis = baseline_prox - baseline_distal
pylon_axis = pylon_axis / np.linalg.norm(pylon_axis)

print("\nPylon axis:")
print(pylon_axis)

# ============================================================
# PERTURBATION TEST
# ============================================================

eps = 0.001  # 1 mm

for joint_name, qpos_id in [
    ("socket_tx", tx_qpos),
    ("socket_ty", ty_qpos),
]:

    current = float(data.qpos[qpos_id])

    data_test = data.replace(
        qpos=data.qpos.at[qpos_id].set(current + eps)
    )

    data_test = mjx.forward(env.sys, data_test)

    pos = np.array(data_test.site_xpos[distal_id])

    delta = pos - baseline_distal

    proj_pylon = np.dot(delta, pylon_axis)

    print("\n==============================")
    print(joint_name)
    print("==============================")
    print("delta world:")
    print(delta)

    print("\nmovement magnitude:")
    print(np.linalg.norm(delta))

    print("\nprojection onto pylon axis:")
    print(proj_pylon)

# ============================================================
# NUMERICAL JACOBIAN
# ============================================================

eps_jac = 1e-4

print("\n")
print("========================================")
print("NUMERICAL JACOBIAN")
print("========================================")

for joint_name, qpos_id in [
    ("socket_tx", tx_qpos),
    ("socket_ty", ty_qpos),
]:

    current = float(data.qpos[qpos_id])

    data_plus = data.replace(
        qpos=data.qpos.at[qpos_id].set(current + eps_jac)
    )

    data_minus = data.replace(
        qpos=data.qpos.at[qpos_id].set(current - eps_jac)
    )

    data_plus = mjx.forward(env.sys, data_plus)
    data_minus = mjx.forward(env.sys, data_minus)

    pos_plus = np.array(data_plus.site_xpos[distal_id])
    pos_minus = np.array(data_minus.site_xpos[distal_id])

    jac = (pos_plus - pos_minus) / (2 * eps_jac)

    pylon_component = np.dot(jac, pylon_axis)

    print("\n--------------------------------")
    print(joint_name)
    print("--------------------------------")

    print("Jacobian d(site_pos)/d(q):")
    print(jac)

    print("\nNorm:")
    print(np.linalg.norm(jac))

    print("\nComponent along pylon axis:")
    print(pylon_component)