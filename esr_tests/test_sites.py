import numpy as np
import mujoco
import matplotlib.pyplot as plt
from loco_mujoco import ImitationFactory

# ------------------------------------------------------------
# 0. Einstellungen
# ------------------------------------------------------------
SITE_NEW = "right_foot_mimic"          # neue Site am Forefoot
SITE_LEFT = "left_foot_mimic"          # Referenz linke Seite
OLD_PARENT_BODY = "calcn_r"            # Body, an dem die Site vorher hing
OLD_LOCAL_POS = None                   # <- alte lokale Position [x, y, z] eintragen,
                                       #    z.B. [-0.05, 0.0, 0.01]; None = ueberspringen
HINGE_JOINT = "ankle_angle_r"
FOREFOOT_BODY = "esr_forefoot_r"
TOE_GEOM = "esr_toe_contact_r"
SHOW_INTERACTIVE_VIEWER = False        # True = zusaetzlich Viewer oeffnen

# ------------------------------------------------------------
# 1. Environment bauen
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
model = env._model


# ------------------------------------------------------------
# 2. Hilfsfunktionen
# ------------------------------------------------------------
def name2id(objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def body_name(body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def site_world_pos(d, site_name):
    sid = name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)
    return None if sid == -1 else d.site_xpos[sid].copy()


def add_marker(scene, pos, rgba, radius=0.012):
    """Zeichnet eine Kugel als Marker in die Szene (nur Visualisierung)."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def old_site_world_pos(d):
    """Weltposition der frueheren Site (lokaler Offset am Calcn)."""
    if OLD_LOCAL_POS is None:
        return None
    bid = name2id(mujoco.mjtObj.mjOBJ_BODY, OLD_PARENT_BODY)
    R = d.xmat[bid].reshape(3, 3)
    return d.xpos[bid] + R @ np.asarray(OLD_LOCAL_POS)


def render_view(d, center_pos, markers, azimuth=90, elevation=-10, dist=0.5,
                width=960, height=720):
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 1
    opt.frame = mujoco.mjtFrame.mjFRAME_SITE   # Site-Koordinatenachsen (RGB)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = center_pos
    cam.distance, cam.azimuth, cam.elevation = dist, azimuth, elevation

    opt.sitegroup[:] = 0
    opt.sitegroup[0] = 1                # nur Gruppe 0 zeigen
    opt.geomgroup[0] = 0                # optional: Skelett-Geoms ausblenden, falls sie stoeren

    with mujoco.Renderer(model, height, width) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        for pos, rgba in markers:
            if pos is not None:
                add_marker(r.scene, pos, rgba)
        return r.render()


# ------------------------------------------------------------
# 3. Existenz der Site pruefen (Welt- UND lokale Position)
# ------------------------------------------------------------
d = mujoco.MjData(model)
mujoco.mj_resetData(model, d)
mujoco.mj_forward(model, d)

site_id = name2id(mujoco.mjtObj.mjOBJ_SITE, SITE_NEW)
if site_id == -1:
    raise RuntimeError(f"Site '{SITE_NEW}' existiert NICHT im Modell!")

print(f"✅ Site '{SITE_NEW}' existiert. ID = {site_id}")
print(f"   → haengt an Body: {body_name(model.site_bodyid[site_id])}")
print(f"   → lokale Pos (model.site_pos, relativ zum Body): {model.site_pos[site_id]}")
print(f"   → Welt-Pos   (d.site_xpos, nach mj_forward):     {d.site_xpos[site_id]}")

print("\nAlle Mimic-Sites im Modell:")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    if name and "mimic" in name:
        print(f"  {name:22s} → Body: {body_name(model.site_bodyid[i]):18s} "
              f"lokal={np.round(model.site_pos[i], 4)}  welt={np.round(d.site_xpos[i], 4)}")

# ------------------------------------------------------------
# 4. Hinge-Rotationstest
# ------------------------------------------------------------
calcn_id = name2id(mujoco.mjtObj.mjOBJ_BODY, OLD_PARENT_BODY)
forefoot_id = name2id(mujoco.mjtObj.mjOBJ_BODY, FOREFOOT_BODY)
hinge_id = name2id(mujoco.mjtObj.mjOBJ_JOINT, HINGE_JOINT)
toe_geom_id = name2id(mujoco.mjtObj.mjOBJ_GEOM, TOE_GEOM)
qpos_adr = model.jnt_qposadr[hinge_id]

if toe_geom_id == -1:
    print(f"\n⚠️ Geom '{TOE_GEOM}' nicht gefunden. Verfuegbare Geoms:")
    print([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)])

print("\nHinge-Rotationstest (Weltpositionen):")
for theta_deg in [-6, 0, 14]:
    d2 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d2)
    d2.qpos[qpos_adr] = np.deg2rad(theta_deg)
    mujoco.mj_forward(model, d2)

    out = [f"θ={theta_deg:>4}°",
           f"calcn_r={np.round(d2.xpos[calcn_id], 4)}",
           f"forefoot_quat={np.round(d2.xquat[forefoot_id], 4)}",
           f"mimic_neu={np.round(site_world_pos(d2, SITE_NEW), 4)}"]
    old = old_site_world_pos(d2)
    if old is not None:
        out.append(f"mimic_alt={np.round(old, 4)}")
    if toe_geom_id != -1:
        out.append(f"toe={np.round(d2.geom_xpos[toe_geom_id], 4)}")
    print("  " + "  ".join(out))

print("\nErwartung:")
print("  • calcn_r und mimic_alt bleiben konstant (haengen nicht am ESR-Forefoot).")
print("  • mimic_neu, forefoot_quat und toe wandern mit dem Hinge-Winkel.")

# ------------------------------------------------------------
# 5. Visueller Vergleich: links vs. rechts (Ruhepose, Hinge = 0)
#    Marker:  magenta = neue Site rechts,  gruen = alte Site rechts (falls OLD_LOCAL_POS gesetzt),
#             blau = Forefoot-Ursprung (Hinge),  gelb = Toe-Geom
# ------------------------------------------------------------
d = mujoco.MjData(model)
mujoco.mj_resetData(model, d)
mujoco.mj_forward(model, d)

MAGENTA, GREEN, BLUE, YELLOW, CYAN = ([1, 0, 1, 1], [0, 1, 0, 1], [0, 0.4, 1, 1],
                                       [1, 1, 0, 1], [0, 1, 1, 1])

right_markers = [
    (site_world_pos(d, SITE_NEW), MAGENTA),
    (old_site_world_pos(d), GREEN),
    (d.xpos[forefoot_id], BLUE),
    (d.geom_xpos[toe_geom_id] if toe_geom_id != -1 else None, YELLOW),
]
left_markers = [(site_world_pos(d, SITE_LEFT), CYAN)]


# einmalig vor dem Rendern: alle Mimic-Sites ausser den beiden Fuss-Sites in eine ausgeblendete Gruppe schieben
keep = {"right_foot_mimic", "left_foot_mimic"}
for i in range(model.nsite):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    model.site_group[i] = 0 if n in keep else 5

model.vis.scale.framelength = 0.2   # Achsen kuerzer
model.vis.scale.framewidth = 0.02   # Achsen duenner

model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 960

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

right_center = site_world_pos(d, SITE_NEW)
axs[0].imshow(render_view(d, right_center, right_markers, azimuth=90))
axs[0].set_title("rechts (seitlich)")

axs[1].imshow(render_view(d, right_center, right_markers, azimuth=180, elevation=-30))
axs[1].set_title("rechts (von hinten/oben)")

left_center = site_world_pos(d, SITE_LEFT)
if left_center is not None:
    axs[2].imshow(render_view(d, left_center, left_markers, azimuth=270))
    axs[2].set_title("links (Referenz)")
else:
    axs[2].text(0.5, 0.5, f"'{SITE_LEFT}' nicht gefunden", ha="center")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Optional: interaktiver Viewer
# ------------------------------------------------------------
if SHOW_INTERACTIVE_VIEWER:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, d) as v:
        v.opt.sitegroup[:] = 1
        v.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        v.cam.lookat[:] = right_center
        v.cam.distance = 0.6
        v.cam.azimuth, v.cam.elevation = 90, -10
        while v.is_running():
            v.sync()

d = mujoco.MjData(model)
mujoco.mj_resetData(model, d)
mujoco.mj_forward(model, d)

fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "esr_forefoot_r")
lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_mimic")

target = d.site_xpos[lid].copy()
target[1] *= -1                       # an der Sagittalebene spiegeln
R = d.xmat[fid].reshape(3, 3)
local_pos = R.T @ (target - d.xpos[fid])
print("pos fuer add_site:", np.round(local_pos, 4))