"""
ESR Prosthesis Sanity Check Script
===================================
Checks masses, positions, contact boxes, and pairs for the ESR prosthesis model.
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
from loco_mujoco import ImitationFactory

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# ── Environment Setup ─────────────────────────────────────────────────────────
env_params = {
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial",
    "amputated_tibia_length": 0.2,
    "reattach_muscles": {"med_gas": [0, 0, 0]},
    "prosthesis_subtype": "ESR",
    "tibia_socket_overlap": 0.2,
    "remove_joint_names": ["subtalar_angle"],
    "ESR_k_lin": 30000.0,
    "ESR_c_theta": 10.0,
    "socket_joint_dofs": [
        'socket_tx', 'socket_ty', 'socket_tz',
        'socket_flexion', 'socket_adduction', 'socket_rotation'
    ],
    "multi_contact_geom_type": "2boxes",
    "contact_geom_solref": [-900, -300],
    "add_sensors": False,
}

env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    **env_params,
    default_dataset_conf=dict(task="walk")
)
print("Environment created successfully.\n")

# ── MuJoCo Data ───────────────────────────────────────────────────────────────
data = mujoco.MjData(env._model)
mujoco.mj_resetData(env._model, data)
mujoco.mj_forward(env._model, data)

# ── IDs ───────────────────────────────────────────────────────────────────────
def get_body(name):   return mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY,  name)
def get_geom(name):   return mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM,  name)
def get_joint(name):  return mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, name)
def get_site(name):   return mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE,  name)

rearfoot_id  = get_body("talus_r")
forefoot_id  = get_body("esr_forefoot_r")
pylon_id     = get_body("pylon_socket_r")
calcn_r_id   = get_body("calcn_r")
toes_r_id    = get_body("toes_r")
calcn_l_id   = get_body("calcn_l")
toes_l_id    = get_body("toes_l")

hinge_id         = get_joint("esr_hinge_r")
rearfoot_geom_id = get_geom("esr_rearfoot_geom_r")
forefoot_geom_id = get_geom("esr_forefoot_geom_r")
pylon_geom_id    = get_geom("pylon_geom")
heel_contact_id  = get_geom("esr_heel_contact_r")
toe_contact_id   = get_geom("esr_toe_contact_r")
rearfoot_com_id  = get_site("esr_rearfoot_COM_r")
forefoot_com_id  = get_site("esr_forefoot_COM_r")


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. MASS CHECK")
print("=" * 60)
print(f"  Hinge joint ID : {hinge_id}  (expected: >= 0)")
print(f"  Rearfoot mass  : {env._model.body_mass[rearfoot_id]:.3f} kg  (expected: 0.356)")
print(f"  Forefoot mass  : {env._model.body_mass[forefoot_id]:.3f} kg  (expected: 0.238)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. BODY WORLD POSITIONS")
print("=" * 60)
print(f"  Pylon socket   : z = {data.xpos[pylon_id][2]*1000:.1f} mm")
print(f"  Rearfoot (talus): z = {data.xpos[rearfoot_id][2]*1000:.1f} mm")
print(f"  Forefoot (FJC) : z = {data.xpos[forefoot_id][2]*1000:.1f} mm  (expected: 27.5 mm)")
print(f"  Floor          : z = 0.0 mm")
print(f"\n  Rearfoot rotation matrix:")
print(f"  {np.round(data.xmat[rearfoot_id].reshape(3,3), 3)}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. GEOM Z-POSITIONS (center, top, bottom)")
print("=" * 60)

def print_geom_z(label, geom_id, size_axis=1):
    center_z = data.geom_xpos[geom_id][2]
    half     = env._model.geom_size[geom_id][size_axis]
    print(f"  {label}:")
    print(f"    center={center_z*1000:.1f}mm  top={( center_z+half)*1000:.1f}mm  bottom={(center_z-half)*1000:.1f}mm")

# Pylon (largest dimension is length)
pylon_size   = env._model.geom_size[pylon_geom_id]
half_pylon   = max(pylon_size)
pylon_center = data.geom_xpos[pylon_geom_id][2]
print(f"  Pylon:")
print(f"    center={pylon_center*1000:.1f}mm  top={(pylon_center+half_pylon)*1000:.1f}mm  bottom={(pylon_center-half_pylon)*1000:.1f}mm")
print(f"    (pylon_socket origin={data.xpos[pylon_id][2]*1000:.1f}mm, gap to socket={( data.xpos[pylon_id][2]-(pylon_center+half_pylon))*1000:.1f}mm)")

print_geom_z("Rearfoot box", rearfoot_geom_id, size_axis=1)
print_geom_z("Forefoot box", forefoot_geom_id, size_axis=1)


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. CENTER OF MASS POSITIONS")
print("=" * 60)
print(f"  Rearfoot COM site : z = {data.site_xpos[rearfoot_com_id][2]*1000:.1f} mm")
print(f"  Forefoot COM site : z = {data.site_xpos[forefoot_com_id][2]*1000:.1f} mm")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. CONTACT BOX CHECK")
print("=" * 60)

# Biological boxes
print("\n  -- Biological boxes (should be deleted or inactive for prosthesis side) --")
for name in ["foot_box_r", "toes_box_r", "foot_box_l", "toes_box_l"]:
    gid = get_geom(name)
    if gid == -1:
        print(f"  {name}: DELETED ✓")
    else:
        ct = env._model.geom_contype[gid]
        ca = env._model.geom_conaffinity[gid]
        print(f"  {name}: EXISTS  contype={ct}  conaffinity={ca}")

# ESR contact boxes
print("\n  -- ESR contact boxes --")
for name, gid in [("esr_heel_contact_r", heel_contact_id),
                  ("esr_toe_contact_r",  toe_contact_id)]:
    if gid == -1:
        print(f"  {name}: NOT FOUND ✗")
    else:
        ct  = env._model.geom_contype[gid]
        ca  = env._model.geom_conaffinity[gid]
        pos = data.geom_xpos[gid]
        rot = data.geom_xmat[gid].reshape(3,3)
        half_y = env._model.geom_size[gid][1]
        bottom = pos + rot @ np.array([0, -half_y, 0])
        print(f"  {name}: FOUND ✓  contype={ct}  conaffinity={ca}")
        print(f"    world center: x={pos[0]*1000:.1f}mm  y={pos[1]*1000:.1f}mm  z={pos[2]*1000:.1f}mm")
        print(f"    bottom edge:  z={bottom[2]*1000:.1f}mm  (expected: 0.0mm)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. CONTACT PAIRS")
print("=" * 60)
for i in range(env._model.npair):
    g1 = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_GEOM, env._model.pair_geom1[i])
    g2 = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_GEOM, env._model.pair_geom2[i])
    print(f"  Pair {i}: {g1} <-> {g2}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. CONTACT BOX POSITION CALCULATION (for tuning)")
print("=" * 60)
R_rearfoot = data.xmat[rearfoot_id].reshape(3,3)
R_forefoot  = data.xmat[forefoot_id].reshape(3,3)
box_half_height = env._model.geom_size[heel_contact_id][1] if heel_contact_id != -1 else 0.008

rearfoot_geom_world = data.geom_xpos[rearfoot_geom_id]
forefoot_geom_world = data.geom_xpos[forefoot_geom_id]

# Heel: 40mm posterior of rearfoot geom center, z = box_half_height
heel_target = np.array([rearfoot_geom_world[0] - 0.04,
                         data.xpos[rearfoot_id][1],
                         box_half_height])
heel_local = R_rearfoot.T @ (heel_target - data.xpos[rearfoot_id])

# Toe: 40mm anterior of forefoot geom center, z = box_half_height
toe_target = np.array([forefoot_geom_world[0] + 0.04,
                        data.xpos[forefoot_id][1],
                        box_half_height])
toe_local = R_forefoot.T @ (toe_target - data.xpos[forefoot_id])

print(f"\n  Heel target (world): x={heel_target[0]*1000:.1f}mm  z={heel_target[2]*1000:.1f}mm")
print(f"  Heel pos (local):    [{heel_local[0]:.4f}, {heel_local[1]:.4f}, {heel_local[2]:.4f}]")

print(f"\n  Toe  target (world): x={toe_target[0]*1000:.1f}mm  z={toe_target[2]*1000:.1f}mm")
print(f"  Toe  pos (local):    [{toe_local[0]:.4f}, {toe_local[1]:.4f}, {toe_local[2]:.4f}]")

# Verification
heel_check = data.xpos[rearfoot_id] + R_rearfoot @ heel_local
toe_check  = data.xpos[forefoot_id] + R_forefoot  @ toe_local
print(f"\n  Verification:")
print(f"  Heel world-Z after transform: {heel_check[2]*1000:.1f}mm  (expected: {box_half_height*1000:.1f}mm)")
print(f"  Toe  world-Z after transform: {toe_check[2]*1000:.1f}mm   (expected: {box_half_height*1000:.1f}mm)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. VIEWER")
print("=" * 60)
print("Opening viewer... (close window to exit)")
with mujoco.viewer.launch_passive(env._model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, -0.085, 0.05]
    viewer.cam.distance  = 1
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = 0
    while viewer.is_running():
        viewer.sync()