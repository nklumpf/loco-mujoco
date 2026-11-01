"""
Achsen-Test fuer --socket_pos / --socket_ori: Tabelle + ein Bild pro Offset.

Der Env wird wie in eval_esr.py gebaut (Offsets ueber domain_randomization_params mit
festen Bereichen [v, v]), mit festem Key resettet und danach --settle_steps Schritte mit
Null-Aktion gefahren (der Offset wird evtl. erst im step wirksam). Baseline und Offset
starten mit identischer qpos, der Unterschied ist also der Alignment-Offset.

Ausgabe:
  1) Tabelle: Verschiebung [mm] und Drehung [deg] der Prothese (talus_r) in anatomischen
     Richtungen (vorne, oben, rechts).
  2) Pro Offset ein PNG (Seite / Front / Oben) mit Overlay:
        ROT   = nur Baseline,   CYAN = nur Offset,   GRAU = identisch.

Aufruf:
    python test_alignment_axes.py --path outputs/<datum>/<zeit>/PPOJax_saved.pkl
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless wie in eval_esr.py

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

# Namen wie in eval_esr.py
POSE_BODY = "talus_r"               # Koerper, dessen Verschiebung/Drehung ausgegeben wird
HEEL_GEOM = "esr_heel_contact_r"    # nur fuer die Richtung "vorne"
TOE_GEOM = "esr_toe_contact_r"

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Pfad zur PPOJax_saved.pkl")
parser.add_argument("--pos_step", type=float, default=0.05, help="Positionsoffset [m]")
parser.add_argument("--ori_step", type=float, default=0.30, help="Orientierungsoffset [rad?]")
parser.add_argument("--settle_steps", type=int, default=1, help="env.step mit Null-Aktion nach dem Reset")
parser.add_argument("--plus_only", action="store_true", help="nur positive Vorzeichen (schneller)")
parser.add_argument("--out", type=str, default="alignment_views", help="Ausgabeordner fuer die Bilder")
parser.add_argument("--dist", type=float, default=1.0, help="Kameraabstand [m]")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

agent_conf, _ = PPOJax.load_agent(args.path)
config = agent_conf.config
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
rand_type = config.randomization_config["randomization_type"]


def reset_pose(socket_pos, socket_ori):
    """Env bauen (wie eval_esr.py), resetten, settle-Schritte, Pose + MjData zurueckgeben."""
    dr_params = {
        "randomize_prosthesis_dof_damping": False,
        "prosthesis_dof_damping_range": {},
        "randomize_prosthesis_joint_stiffness": False,
        "prosthesis_joint_stiffness_range": {},
        "randomize_prosthesis_body_position": True,
        "prosthesis_body_position_range": {
            "pylon_socket": {"x": [socket_pos[0]] * 2, "z": [socket_pos[1]] * 2},
            "talus": {"x": [0.0, 0.0], "z": [0.0, 0.0]},
        },
        "randomize_prosthesis_body_orientation": True,
        "prosthesis_body_orientation_range": {
            "pylon_socket": {"x": [socket_ori[0]] * 2, "y": [socket_ori[1]] * 2, "z": [socket_ori[2]] * 2},
            "talus": {"x": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    }
    env = factory.make(
        **config.experiment.env_params,
        **config.experiment.task_factory.params,
        domain_randomization_type=rand_type,
        domain_randomization_params=dr_params)
    mj_model = env._model
    env = PPOJax._wrap_env(env, config.experiment)

    keys = jax.random.split(jax.random.key(0), 2)[1:]   # gleicher Key wie eval_esr.py
    _, state = env.reset(keys)

    try:
        n_act = int(env.info.action_space.shape[0])
    except Exception:
        n_act = int(mj_model.nu)
    for _ in range(args.settle_steps):
        *_, state = env.step(state, jnp.zeros((1, n_act)))

    # Batch-Dimension (1) entfernen und in ein normales MjData kopieren (fuer Auslesen + Rendern)
    data0 = jax.tree.map(lambda x: x[0] if hasattr(x, "ndim") and x.ndim > 0 else x, state.data)
    d = mujoco.MjData(mj_model)
    mjx.get_data_into(d, mj_model, data0)

    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, POSE_BODY)
    hid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, HEEL_GEOM)
    tid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, TOE_GEOM)
    return {
        "model": mj_model,
        "data": d,
        "pos": np.array(d.xpos[bid]),
        "mat": np.array(d.xmat[bid]).reshape(3, 3),
        "heel": np.array(d.geom_xpos[hid]),
        "toe": np.array(d.geom_xpos[tid]),
        "up": -np.asarray(mj_model.opt.gravity) / np.linalg.norm(mj_model.opt.gravity),
    }


def anatomical_frame(base):
    """Zeilen: vorne, oben, rechts (Weltvektoren), aus Gravitation und Ferse->Zeh der Baseline."""
    up = base["up"]
    fwd = base["toe"] - base["heel"]
    fwd = fwd - up * np.dot(fwd, up)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)          # vorne x oben = rechts
    return np.stack([fwd, up, right])


def az_of(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


class Viewer:
    """Rendert Seite / Front / Oben um die Prothese herum (setzt Welt-z als oben voraus)."""

    def __init__(self, base, F):
        if abs(base["up"][2]) < 0.99:
            raise RuntimeError("Welt-'oben' ist nicht die z-Achse -> Kamerawinkel in Viewer anpassen.")
        fwd, _, right = F
        self.lookat = base["pos"] + base["up"] * 0.15
        # (Name, Azimut, Elevation) -- Blickrichtung der Kamera
        self.views = [("Seite (von rechts)", az_of(-right), 0.0),
                      ("Front (von vorne)", az_of(-fwd), 0.0),
                      ("Oben (Zeh = oben im Bild)", az_of(fwd), -89.0)]
        self.renderer = mujoco.Renderer(base["model"], height=480, width=640)
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False   # Muskel-Clutter ausblenden

    def render(self, d):
        imgs = []
        for _, az, el in self.views:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = self.lookat
            cam.distance = args.dist
            cam.azimuth, cam.elevation = az, el
            self.renderer.update_scene(d, camera=cam, scene_option=self.opt)
            imgs.append(self.renderer.render().copy())
        return imgs


def overlay(img_base, img_off):
    gb, go = img_base.mean(axis=2), img_off.mean(axis=2)
    out = np.stack([gb, go, go], axis=2)           # Baseline -> rot, Offset -> cyan
    return np.clip(out, 0, 255).astype(np.uint8)


def measure(base, out, F):
    d = F @ (out["pos"] - base["pos"]) * 1000.0
    rot = F @ R.from_matrix(out["mat"] @ base["mat"].T).as_rotvec() * 180 / np.pi
    return d, rot


if __name__ == "__main__":
    zero2, zero3 = [0.0, 0.0], [0.0, 0.0, 0.0]
    base = reset_pose(zero2, zero3)
    F = anatomical_frame(base)
    viewer = Viewer(base, F)
    base_imgs = viewer.render(base["data"])
    signs = (+1,) if args.plus_only else (+1, -1)

    cases = []
    for idx, ax in ((0, "x"), (1, "z")):            # socket_pos hat nur x und z
        for s in signs:
            pos = list(zero2)
            pos[idx] = s * args.pos_step
            cases.append((f"pos {ax} {s * args.pos_step:+.2f}", pos, zero3))
    for idx, ax in ((0, "x"), (1, "y"), (2, "z")):
        for s in signs:
            ori = list(zero3)
            ori[idx] = s * args.ori_step
            cases.append((f"ori {ax} {s * args.ori_step:+.2f}", zero2, ori))

    print(f"{POSE_BODY}: Verschiebung [mm] und Drehung [deg], jeweils (vorne, oben, rechts)")
    print(f"{'Offset':<16}{'Verschiebung [mm]':<34}{'Drehung [deg]'}")
    for label, pos, ori in cases:
        out = reset_pose(pos, ori)
        d, rot = measure(base, out, F)
        print(f"{label:<16}{np.array2string(d, precision=1, suppress_small=True):<34}"
              f"{np.array2string(rot, precision=1, suppress_small=True)}")

        imgs = viewer.render(out["data"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for a, (name, _, _), ib, io in zip(axes, viewer.views, base_imgs, imgs):
            a.imshow(overlay(ib, io))
            a.set_title(name)
            a.axis("off")
        fig.suptitle(f"{label}   |   rot = Baseline, cyan = Offset   |   "
                     f"Verschiebung [mm] (v,o,r): {np.round(d, 1)}   Drehung [deg] (v,o,r): {np.round(rot, 1)}",
                     fontsize=9)
        fig.tight_layout()
        fname = label.replace(" ", "_").replace("+", "p").replace("-", "m") + ".png"
        fig.savefig(os.path.join(args.out, fname), dpi=120)
        plt.close(fig)
    print(f"\nBilder gespeichert in: {os.path.abspath(args.out)}")