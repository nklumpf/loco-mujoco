import os
import mujoco
import mujoco.viewer
import numpy as np
from loco_mujoco import ImitationFactory

os.environ['XLA_FLAGS'] = ('--xla_gpu_triton_gemm_any=True')

env_params_sach = {
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial",
    "amputated_tibia_length": 0.2,
    "reattach_muscles": {"med_gas": [0, 0, 0]},
    "prosthesis_subtype": "SACH",
    "tibia_socket_overlap": 0.2,
    "remove_joint_names": ["subtalar_angle"],
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
    **env_params_sach,
    default_dataset_conf=dict(task="walk")
)
print("SACH Environment created!")

data = mujoco.MjData(env._model)
mujoco.mj_resetData(env._model, data)
mujoco.mj_forward(env._model, data)

# IDs
pylon_body_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, "pylon_socket_r")
talus_id      = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, "talus_r")
pylon_geom_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "pylon_geom")
socket_geom_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "socket_geom")
sach_geom_id  = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "sach_geom")
foot_box_id   = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "foot_box_r")
toes_box_id   = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "toes_box_r")

pylon_size  = env._model.geom_size[pylon_geom_id]
socket_size = env._model.geom_size[socket_geom_id]

print(f"\n=== SACH: Z-POSITIONEN ===")
print(f"pylon_socket_r origin:   z = {data.xpos[pylon_body_id][2]*1000:.1f} mm")
print(f"talus_r origin:          z = {data.xpos[talus_id][2]*1000:.1f} mm")

print(f"\n-- Socket Geom --")
print(f"  Geom center (z):   {data.geom_xpos[socket_geom_id][2]*1000:.1f} mm")
print(f"  half-size:         {socket_size*1000}")
print(f"  Oberkante (z):     {(data.geom_xpos[socket_geom_id][2] + max(socket_size))*1000:.1f} mm")
print(f"  Unterkante (z):    {(data.geom_xpos[socket_geom_id][2] - max(socket_size))*1000:.1f} mm")

print(f"\n-- Pylon Geom --")
print(f"  Geom center (z):   {data.geom_xpos[pylon_geom_id][2]*1000:.1f} mm")
print(f"  half-size:         {pylon_size*1000}")
half_pylon = max(pylon_size)
print(f"  Oberkante (z):     {(data.geom_xpos[pylon_geom_id][2] + half_pylon)*1000:.1f} mm")
print(f"  Unterkante (z):    {(data.geom_xpos[pylon_geom_id][2] - half_pylon)*1000:.1f} mm")

if sach_geom_id != -1:
    sach_size = env._model.geom_size[sach_geom_id]
    print(f"\n-- SACH Foot Geom --")
    print(f"  Geom center (z):   {data.geom_xpos[sach_geom_id][2]*1000:.1f} mm")
    print(f"  half-size:         {sach_size*1000}")
    print(f"  Oberkante (z):     {(data.geom_xpos[sach_geom_id][2] + max(sach_size))*1000:.1f} mm")
    print(f"  Unterkante (z):    {(data.geom_xpos[sach_geom_id][2] - max(sach_size))*1000:.1f} mm")

print(f"\n-- Kontaktboxen --")
print(f"  foot_box_r (z):    {data.geom_xpos[foot_box_id][2]*1000:.1f} mm")
print(f"  toes_box_r (z):    {data.geom_xpos[toes_box_id][2]*1000:.1f} mm")

print(f"\n=== ZUSAMMENFASSUNG SACH ===")
print(f"Socket Ursprung (oben):  z = {data.xpos[pylon_body_id][2]*1000:.1f} mm")
print(f"Pylon Oberkante:         z = {(data.geom_xpos[pylon_geom_id][2] + half_pylon)*1000:.1f} mm")
print(f"Pylon Unterkante:        z = {(data.geom_xpos[pylon_geom_id][2] - half_pylon)*1000:.1f} mm")
print(f"Talus origin:            z = {data.xpos[talus_id][2]*1000:.1f} mm")
print(f"Boden:                   z = 0.0 mm")
print(f"\n→ Lücke Pylon-Unterkante zu Talus: {(data.geom_xpos[pylon_geom_id][2] - half_pylon - data.xpos[talus_id][2])*1000:.1f} mm")
print(f"→ Lücke Pylon-Oberkante zu Socket:  {(data.xpos[pylon_body_id][2] - (data.geom_xpos[pylon_geom_id][2] + half_pylon))*1000:.1f} mm")

print("\nOpening viewer...")
with mujoco.viewer.launch_passive(env._model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, -0.085, 0.05]
    viewer.cam.distance = 1
    viewer.cam.azimuth = 90
    viewer.cam.elevation = 0
    while viewer.is_running():
        viewer.sync()