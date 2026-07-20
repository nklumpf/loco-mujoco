import numpy as np
import mujoco
from loco_mujoco import ImitationFactory

# ------------------------------------------------------------
# 1. Build ESR environment
# ------------------------------------------------------------
env_params = {
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial",
    "prosthesis_subtype": "ESR",
    "ESR_model_type": "linear",
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2,
    "reattach_muscles": {"med_gas": [0, 0, 0]},
    "remove_joint_names": ["subtalar_angle"],
}

print("Creating ESR environment...")
env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    **env_params,
    default_dataset_conf=dict(task="walk"),
)

model = env._model  # MuJoCo model

# ------------------------------------------------------------
# 2. Check if right_foot_mimic exists
# ------------------------------------------------------------
site_name = "right_foot_mimic"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

if site_id == -1:
    print(f"❌ Site '{site_name}' does NOT exist in the ESR model!")
else:
    print(f"✅ Site '{site_name}' exists. ID = {site_id}")

    # Body to which the site is attached
    body_id = model.site_bodyid[site_id]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    print(f"   → Site is attached to body: {body_name}")

    # Absolute world position of the site
    pos = model.site_pos[site_id]
    print(f"   → Site world position (model.site_pos): {pos}")

# ------------------------------------------------------------
# 3. List all mimic sites
# ------------------------------------------------------------
print("\nAll mimic sites in the model:")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    if "mimic" in name:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[i])
        print(f"  {name:20s}  → Body: {body}")

# ------------------------------------------------------------
# 4. ESR hinge rotation test (corrected)
#    Important:
#      - The forefoot body origin lies exactly at the hinge joint.
#        Therefore its world position (xpos) does NOT change under pure rotation.
#      - To observe hinge rotation, we must inspect:
#           * forefoot orientation (xquat)
#           * world position of a geom far from the hinge
#             (e.g., esr_toe_contact_r)
# ------------------------------------------------------------
print("\nTesting hinge rotation effect (corrected: geom world pos + orientation):")

d = mujoco.MjData(model)
mujoco.mj_resetData(model, d)

calcn_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
forefoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "esr_forefoot_r")
hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
qpos_adr = model.jnt_qposadr[hinge_id]

toe_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "esr_toe_contact_r")
if toe_geom_id == -1:
    print("⚠️  Geom 'esr_toe_contact_r' not found — check geom names via:")
    print("    [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]")

for theta_deg in [-6, 0, 14]:
    d.qpos[qpos_adr] = np.deg2rad(theta_deg)
    mujoco.mj_forward(model, d)

    calcn_pos = d.xpos[calcn_id]
    forefoot_quat = d.xquat[forefoot_id]

    line = (f"θ={theta_deg:>4}°  "
            f"calcn_r pos={np.round(calcn_pos, 4)}  "
            f"forefoot quat={np.round(forefoot_quat, 4)}")

    if toe_geom_id != -1:
        toe_world_pos = d.geom_xpos[toe_geom_id]
        line += f"  toe_contact world pos={np.round(toe_world_pos, 4)}"

    print(line)

print("\nExpectation:")
print("  • calcn_r pos should remain constant (it is NOT the ESR foot).")
print("  • forefoot orientation (xquat) and toe_contact world position")
print("    SHOULD change with hinge rotation.")
print("If forefoot/geom positions change but right_foot_mimic stays on calcn_r,")
print("it proves that the mimic site does NOT capture ESR hinge rotation.")
