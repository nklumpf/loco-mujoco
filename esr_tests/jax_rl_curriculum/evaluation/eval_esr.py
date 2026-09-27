"""
This script evaluates a trained PPOJax locomotion policy using the MJX simulation and analyses the motion and joint dynamics of the prosthetic and intact legs, for both ESR and SACH prosthesis subtypes.

During the evaluation, the following quantities are recorded externally from the MJX simulation state, for hip / knee / ankle, left and right:
    - Joint angle (qpos)
    - Joint angular velocity (qvel)
    - Joint moment (mimic torque sensors)
    - Reference (GoalTrajMimicv2) joint angle, where available

Joint powers are calculated afterwards from the corresponding joint moment and angular velocity:
    P = M * q_dot

For ESR, the right "ankle" is represented by the ESR hinge joint (esr_hinge_r) instead of a real ankle_angle_r joint (which does not exist in the ESR model). To keep the exported CSV columns identical between ESR and SACH runs, the ORIGINAL ankle_right_moment (right_foot_mimic-based) is still exported unchanged for backward compatibility; the corrected version is exported under separate column names (see CSV export section).

The observed ESR range of motion (ROM) is calculated and compared with the reference ROM reported by Lecomte et al.

The script uses the MJX evaluation path and must therefore be run without the --use_mujoco option.

The --record option can optionally be used to record the MJX rendering.

Example:
    python eval_esr.py --path path/to/PPOJax_saved.pkl --n_steps 1000 --record --body_mass_kg 86.6
"""

import os
import re
import datetime

# Uncomment the following lines to force JAX to use the CPU instead of GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax

# Uncomment the following line to force JAX to use the CPU instead of GPU.
# jax.config.update("jax_platform_name", "cpu")

import argparse

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf


os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

# headless
os.environ["MUJOCO_GL"] = "egl"


# =============================================================================
# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate PPOJax agent and analyse prosthesis gait (ESR or SACH).")
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--n_steps', type=int, default=1000, help='Number of evaluation steps')
parser.add_argument('--record', action='store_true', help='Record MJX rendering')
parser.add_argument('--body_mass_kg', type=float, default=None,
                     help='Body mass in kg. Used for the GRF sanity check AND to '
                          'normalize moment/power (Nm/kg, W/kg) for literature comparison '
                          'and for the gait-cycle heel-strike detection threshold.')
args = parser.parse_args()


def extract_run_timestamp(checkpoint_path):
    """
    Extract a date_time identifier from a checkpoint path shaped like
    'outputs/2026-09-12/07-15-31/PPOJax_saved.pkl' -> '2026-09-12/07-15-31'.
    Falls back to the current time if the expected pattern isn't found.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})/(\d{2}-\d{2}-\d{2})", str(checkpoint_path))
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    fallback = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    print(f"WARNING: could not parse date/time from checkpoint path "
          f"'{checkpoint_path}', falling back to current time '{fallback}'.")
    return fallback


run_timestamp = extract_run_timestamp(args.path)
print(f"Run timestamp for output folder: {run_timestamp}")


# =============================================================================
# Load trained agent
print("\nLoading trained agent...")
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
domain_randomization_type = config.randomization_config["randomization_type"]
domain_randomization_params = {
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {},
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {
        "pylon_socket": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]},
        "talus": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]},
    },
    "randomize_prosthesis_body_orientation": True,
    "prosthesis_body_orientation_range": {
        "pylon_socket": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
        "talus": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]},
    },
}

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"  # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True

print("\nCreating environment...")
env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    domain_randomization_type=domain_randomization_type,
    domain_randomization_params=domain_randomization_params)

prosthesis_subtype = config.experiment.env_params["prosthesis_subtype"]
esr_model_type     = config.experiment.env_params.get("ESR_model_type", None)
is_esr  = (prosthesis_subtype == "ESR")
is_sach = (prosthesis_subtype == "SACH")


def find_prosthesis_config(obj):
    """
    Recursively search env wrappers to locate the actual ESR prosthesis config object, identified by ESR-specific attributes.
    """
    visited = set()

    def search(x):
        """
        Helper: depth-first search through wrapper attributes to find an object containing ESR parameters.
        """
        if x is None or id(x) in visited:
            return None
        visited.add(id(x))
        if hasattr(x, "ESR_lever_arm") and hasattr(x, "ESR_hinge_base_stiffness"):
            return x
        for name in ["env", "_env", "wrapped_env"]:
            if hasattr(x, name):
                result = search(getattr(x, name))
                if result is not None:
                    return result
        return None

    return search(obj)


if is_esr:
    prosthesis_cfg_obj = find_prosthesis_config(env)
    if prosthesis_cfg_obj is None:
        print("WARNING: could not locate the prosthesis config object on the env "
              "wrapper chain -- falling back to class defaults for ESR torque "
              "reconstruction (lever arm, stiffness, linear/nonlinear params). "
              "Double-check these match your actual training config!")

    def _cfg(name, default):
        """
        Read parameter from prosthesis config object or fall back to provided default if not found.
        """
        return getattr(prosthesis_cfg_obj, name, default) if prosthesis_cfg_obj is not None else default

    ESR_LEVER_ARM = _cfg("ESR_lever_arm", 0.128)
    ESR_HINGE_BASE_STIFFNESS = _cfg("ESR_hinge_base_stiffness", 250.0)
    ESR_MODEL_TYPE_RESOLVED = _cfg("ESR_model_type", "linear")
    LINEAR_PARAMS = _cfg("linear_params", {"k_heel": 57507.3, "k_keel": 24103.6})
    NONLINEAR_PARAMS = _cfg("nonlinear_params", {
        "heel": {"a": 1781838.2, "b": 32644.4},
        "keel": {"a": 172171.9, "b": 18212.8}})
    print(f"\nResolved ESR params for torque reconstruction: "
          f"lever_arm={ESR_LEVER_ARM}, base_stiffness={ESR_HINGE_BASE_STIFFNESS}, "
          f"model_type={ESR_MODEL_TYPE_RESOLVED}")


def compute_esr_diagnostic_torques(theta_rad, esr_model_type, lever_arm, base_stiffness,
                                    linear_params, nonlinear_params):
    """
    Post-hoc NumPy reconstruction of compute_ESR_hinge_qfrc(), applied to an already-logged theta array (radians). This is a pure function of theta and fixed parameters, so it can be computed after the simulation loop instead of instrumenting the JAX pre-step function.

    Returns:
        tau_ESR            -- desired total ESR joint torque (Lecomte force law)
        tau_mujoco_spring  -- torque already contributed by MuJoCo's native hinge spring 
        tau_applied        -- the correction actually injected via qfrc_applied 
                              (tau_ESR - tau_mujoco_spring)
    """
    theta = np.asarray(theta_rad, dtype=float)
    theta_clipped = np.clip(theta, np.deg2rad(-6.0), np.deg2rad(14.0))
    z = lever_arm * np.sin(theta_clipped)

    if esr_model_type == "linear":
        k = np.where(theta_clipped < 0, linear_params["k_heel"], linear_params["k_keel"])
        F_z = k * z
    elif esr_model_type == "nonlinear":
        a = np.where(theta_clipped < 0, nonlinear_params["heel"]["a"], nonlinear_params["keel"]["a"])
        b = np.where(theta_clipped < 0, nonlinear_params["heel"]["b"], nonlinear_params["keel"]["b"])
        F_z = (a * np.abs(z) + b) * z
    else:
        raise ValueError(f"Unknown ESR_model_type: {esr_model_type}")

    tau_ESR = -F_z * lever_arm * np.cos(theta_clipped)
    tau_mujoco_spring = -base_stiffness * theta
    tau_applied = tau_ESR - tau_mujoco_spring
    return tau_ESR, tau_mujoco_spring, tau_applied


def detect_heel_strikes(grf, threshold_n=50.0, min_stride_samples=20):
    """
    Detect rising-edge crossings of 'grf' above 'threshold_n' as heel-strike events (start of stance). A minimum sample spacing filters out noise-triggered double detections around the threshold.
    """
    grf = np.asarray(grf)
    is_stance = grf > threshold_n
    crossings = np.where(np.diff(is_stance.astype(int)) == 1)[0] + 1
    filtered = []
    for idx in crossings:
        if not filtered or (idx - filtered[-1]) >= min_stride_samples:
            filtered.append(idx)
    return np.asarray(filtered, dtype=int)


def resample_strides_to_gait_cycle(signal, strike_indices, n_points=101):
    """
    Cut 'signal' into strides between consecutive heel-strike indices and resample each stride onto a common 0-100% gait-cycle grid via linear interpolation. Returns an array of shape (n_strides, n_points).
    """
    signal = np.asarray(signal)
    curves = []
    for i in range(len(strike_indices) - 1):
        start, end = int(strike_indices[i]), int(strike_indices[i + 1])
        if end - start < 5:
            continue
        stride = signal[start:end]
        x_orig = np.linspace(0, 100, num=len(stride))
        x_new = np.linspace(0, 100, num=n_points)
        curves.append(np.interp(x_new, x_orig, stride))
    return np.asarray(curves) if curves else np.empty((0, n_points))


# =============================================================================
# Find underlying MuJoCo model
def find_mujoco_model(obj):
    """
    Recursively search through the environment wrappers for the underlying MuJoCo model.
    """
    visited = set()

    def search(x):
        if x is None:
            return None
        if id(x) in visited:
            return None
        visited.add(id(x))

        for name in ["mj_model", "model", "_model"]:
            if hasattr(x, name):
                candidate = getattr(x, name)
                if candidate is None:
                    continue
                if hasattr(candidate, "jnt_qposadr") and hasattr(candidate, "joint"):
                    return candidate

        for name in ["env", "_env", "wrapped_env"]:
            if hasattr(x, name):
                child = getattr(x, name)
                result = search(child)
                if result is not None:
                    return result
        return None

    return search(obj)


mj_model = find_mujoco_model(env)
if mj_model is None:
    raise RuntimeError("Could not find the underlying MuJoCo model.")
print("\nMuJoCo model found.")

# =============================================================================
# CHECKS
#=== check mass of ESR-prosthesis ===
if is_esr:
    print("\n" + "=" * 60)
    print("BODY MASS CHECK (ESR rearfoot subtree) — checking for double-counted mass")
    for name in ["pylon_socket_r", "talus_r", "calcn_r", "toes_r", "esr_forefoot_r"]:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            print(f"  {name:20s}: not found in model")
            continue
        print(f"  {name:20s}: own mass = {mj_model.body_mass[bid]:.4f} kg, "
              f"subtree mass = {mj_model.body_subtreemass[bid]:.4f} kg")
    # Expected masses
    expected_rearfoot_mass = getattr(prosthesis_cfg_obj, "ESR_rearfoot_mass", 0.356)
    expected_forefoot_mass = getattr(prosthesis_cfg_obj, "ESR_forefoot_mass", 0.238)
    print(f"\n  Expected from config: rearfoot = {expected_rearfoot_mass:.4f} kg, "
          f"forefoot = {expected_forefoot_mass:.4f} kg")

    leftover_mass = 0.0
    for name in ["calcn_r", "toes_r"]:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            leftover_mass += mj_model.body_mass[bid]
    if leftover_mass > 1e-3:
        print(f"\n  WARNING: calcn_r + toes_r still carry {leftover_mass:.4f} kg of "
              f"leftover BIOLOGICAL mass on top of the intended ESR masses. This "
              f"inflates inertial contributions at the prosthetic_ankle_r sensor. "
              f"Fix: zero their mass/inertia for ESR in add_prosthesis_properties(), "
              f"e.g. body.mass = 0.0; body.fullinertia = [0]*6 for calcn_r/toes_r "
              f"when self.prosthesis_subtype == 'ESR'.")
    else:
        print(f"\n  OK: calcn_r + toes_r carry ~0 kg — no leftover biological mass detected.")


# === All Joints ===
print("\nJoints in MuJoCo model:")
for i in range(mj_model.njnt):
    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_adr = int(mj_model.jnt_qposadr[i])
    qvel_adr = int(mj_model.jnt_dofadr[i])
    print(f"  ID {i:2d}: {joint_name:30s} qpos={qpos_adr:2d}, qvel={qvel_adr:2d}")

# === All Sensors ===
print("\nSensors in MuJoCo model:")
for i in range(mj_model.nsensor):
    sensor_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f"  ID {i:2d}: {sensor_name:40s} "
          f"adr={int(mj_model.sensor_adr[i])}, dim={int(mj_model.sensor_dim[i])}")

# =============================================================================
# Helper functions for joints, sensors, bodies
def get_joint_qpos_address(model, joint_name):
    """
    Return qpos index of a joint; raise if joint name is not found.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Could not find joint '{joint_name}' in the MuJoCo model.")
    return int(model.jnt_qposadr[joint_id])


def get_joint_qvel_address(model, joint_name):
    """
    Return qvel index (DOF address) of a joint; raise if not found.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Could not find joint '{joint_name}' in the MuJoCo model.")
    return int(model.jnt_dofadr[joint_id])


def get_sensor_id(model, sensor_name):
    """
    Return sensor ID by name; raise if sensor does not exist.
    """
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise RuntimeError(f"Could not find sensor '{sensor_name}' in the MuJoCo model.")
    return int(sensor_id)


def get_sensor_adr(model, sensor_id):
    """
    Return data address of a sensor given its ID.
    """
    return int(model.sensor_adr[sensor_id])


def get_body_id(model, body_name):
    """
    Return body ID by name; raise if body is not found.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body '{body_name}' in the MuJoCo model.")
    return int(body_id)


def try_get_sensor_adr(model, sensor_name):
    """
    Safe sensor lookup: return address or None if sensor is missing.
    """
    try:
        return get_sensor_adr(model, get_sensor_id(model, sensor_name))
    except RuntimeError:
        return None


# =============================================================================
# Joint + torque sensor config (sagittal moment = +2 offset in 3D torque sensor).
# "ankle_right" is a real joint only for SACH; ESR fills this separately for
# consistent CSV columns across prosthesis types.
JOINTS_CONFIG = {
    "hip_left":   {"joint": "hip_flexion_l", "torque_sensor": "left_hip_mimic_torque_sensor"},
    "hip_right":  {"joint": "hip_flexion_r", "torque_sensor": "right_hip_mimic_torque_sensor"},
    "knee_left":  {"joint": "knee_angle_l",  "torque_sensor": "left_knee_mimic_torque_sensor"},
    "knee_right": {"joint": "knee_angle_r",  "torque_sensor": "right_knee_mimic_torque_sensor"},
    "ankle_left": {"joint": "ankle_angle_l", "torque_sensor": "left_foot_mimic_torque_sensor"}}
JOINTS_CONFIG["ankle_right"] = {"joint": "ankle_angle_r", "torque_sensor": "right_foot_mimic_torque_sensor"}

# Resolve qpos/qvel + torque sensor addresses
joint_info = {}
for key, cfg in JOINTS_CONFIG.items():
    joint_info[key] = {
        "qpos_adr": get_joint_qpos_address(mj_model, cfg["joint"]),
        "qvel_adr": get_joint_qvel_address(mj_model, cfg["joint"]),
        "torque_adr": get_sensor_adr(mj_model, get_sensor_id(mj_model, cfg["torque_sensor"]))}

# Unified right-foot torque sensor (used for both ESR/SACH for compatibility)
right_foot_mimic_torque_adr = get_sensor_adr(
    mj_model, get_sensor_id(mj_model, "right_foot_mimic_torque_sensor"))

# ESR-specific sensors
if is_esr:
    esr_hinge_qpos_adr = get_joint_qpos_address(mj_model, "ankle_angle_r")
    esr_hinge_qvel_adr = get_joint_qvel_address(mj_model, "ankle_angle_r")
    esr_hinge_torque_adr = get_sensor_adr(mj_model, get_sensor_id(mj_model, "esr_hinge_torque_r"))

    # Socket flexion DOF (offset; used for corrected ankle angle)
    socket_flexion_qpos_adr = get_joint_qpos_address(mj_model, "socket_flexion_r")
    socket_flexion_qvel_adr = get_joint_qvel_address(mj_model, "socket_flexion_r")

    # Total pylon-to-socket moment (different from hinge torque)
    pylon_torque_adr = try_get_sensor_adr(mj_model, "pylon_mimic_r_torque_sensor")
    if pylon_torque_adr is None:
        print("NOTE: 'pylon_mimic_r_torque_sensor' not found -- esr_pylon_moment "
              "will be NaN in the output.")

    # Corrected ankle moment sensor at pylon–rearfoot attachment
    prosthetic_ankle_torque_adr = try_get_sensor_adr(mj_model, "prosthetic_ankle_r_torque_sensor")
    if prosthetic_ankle_torque_adr is None:
        print("NOTE: 'prosthetic_ankle_r_torque_sensor' not found in this model -- "
              "falling back to right_foot_mimic_torque_sensor for the corrected "
              "ankle_right moment, with a warning at export time. Add the new "
              "site+sensor to MjxSkeletonMuscleProsthesis and re-train/re-export "
              "the model to get the physically correct ankle moment location.")

# -------------------------------------------------------------------------
# Ground reaction force bodies (heel + forefoot per side; summed for total GRF)
if is_esr:
    right_bodies = ["talus_r", "esr_forefoot_r"]
else:
    right_bodies = ["toes_r", "calcn_r"]

GRF_BODY_NAMES = {"left": ["toes_l", "calcn_l"], "right": right_bodies}
VERTICAL_FORCE_INDEX = 5

grf_body_ids = {
    side: [get_body_id(mj_model, name) for name in names]
    for side, names in GRF_BODY_NAMES.items()}


def collect_grf_vertical(data, body_ids, vertical_index=VERTICAL_FORCE_INDEX, env_idx=0):
    """
    Sum the vertical component of cfrc_ext (external force on each body,
    world frame) across the given body ids, for a single env index.
    """
    total = 0.0
    for body_id in body_ids:
        total += float(data.cfrc_ext[env_idx, body_id, vertical_index])
    return total


# Position tracking for step length
POSITION_BODY_NAMES = {"pelvis": "pelvis", "foot_left": "calcn_l", "foot_right": right_bodies[0]}
position_body_ids = {key: get_body_id(mj_model, name) for key, name in POSITION_BODY_NAMES.items()}


# =============================================================================
# Wrap env, set up policy call
env = PPOJax._wrap_env(env, config.experiment)
train_state = agent_state.train_state

if config.experiment.n_seeds > 1:
    train_state = jax.tree.map(lambda x: x[0], train_state)

n_envs = 1
rng = jax.random.key(0)
keys = jax.random.split(rng, n_envs + 1)
rng = keys[0]
env_keys = keys[1:]


def sample_actions(ts, obs, _rng):
    """
    Compute an action from the trained PPO policy. Running observation
    statistics are updated the same way as in PPOJax.play_policy().
    """
    y, updates = agent_conf.network.apply(
        {"params": ts.params, "run_stats": ts.run_stats},
        obs,
        mutable=["run_stats"])
    ts = ts.replace(run_stats=updates["run_stats"])
    pi, _ = y
    action = pi.sample(seed=_rng)
    return action, ts


print("\nJIT compiling policy...")
plcy_call = jax.jit(sample_actions)

obs, env_state = env.reset(env_keys)

# Time Step
sim_dt = mj_model.opt.timestep
print(f"MuJoCo physics timestep: {sim_dt} s")
# =============================================================================
# Initialize evaluation data (generic, per joint key)
steps = []
sim_time_log = [0.0]

logs = {key: {"angle": [], "vel": [], "moment": []} for key in joint_info}
ref_logs = {key: [] for key in joint_info}

right_foot_mimic_moment_log = []

esr_hinge_log = {"angle": [], "vel": [], "moment": []} if is_esr else None
socket_flexion_log = [] if is_esr else None
socket_flexion_qvel_log = [] if is_esr else None          # NEW
pylon_moment_log = [] if is_esr else None
prosthetic_ankle_moment_log = [] if is_esr else None       # NEW

grf_left_log = []
grf_right_log = []
pelvis_pos_log = []
foot_left_pos_log = []
foot_right_pos_log = []


# =============================================================================
# Evaluation loop
print("\n" + "=" * 60)
print("\nStarting evaluation...")

for i in range(args.n_steps):
    rng, _rng = jax.random.split(rng)

    action, train_state = plcy_call(train_state, obs, _rng)
    action = jnp.atleast_2d(action)

    (obs, reward, absorbing, done, info, env_state) = env.step(env_state, action)

    qpos = env_state.data.qpos[0]
    qvel = env_state.data.qvel[0]
    sensordata = env_state.data.sensordata[0]

    # Reference trajectory (GoalTrajMimicv2)
    traj_state = env_state.additional_carry.traj_state
    traj_sample = env.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no, np)
    ref_qpos = traj_sample.qpos

    steps.append(i)
    sim_time_log.append(float(env_state.data.time[0]))

    # Generic joint logging (hip/knee both sides, ankle_left always, ankle_right if SACH)
    REF_SIGN_FLIP_PREFIXES = ("hip", "knee")

    for key, info_ in joint_info.items():
        logs[key]["angle"].append(float(qpos[info_["qpos_adr"]]))
        logs[key]["vel"].append(float(qvel[info_["qvel_adr"]]))
        logs[key]["moment"].append(float(sensordata[info_["torque_adr"] + 2]))

        ref_val = float(ref_qpos[info_["qpos_adr"]])
        if key.startswith(REF_SIGN_FLIP_PREFIXES):
            ref_val = -ref_val
        ref_logs[key].append(ref_val)

    # right_foot_mimic torque -- always logged, used as unified ankle_right moment
    right_foot_mimic_moment_log.append(float(sensordata[right_foot_mimic_torque_adr + 2]))

    # ESR-specific hinge sensors (diagnostic, not used as ankle_right by default)
    if is_esr:
        esr_hinge_log["angle"].append(float(qpos[esr_hinge_qpos_adr]))
        esr_hinge_log["vel"].append(float(qvel[esr_hinge_qvel_adr]))
        esr_hinge_log["moment"].append(float(sensordata[esr_hinge_torque_adr]))  # dim=1, no +2

        socket_flexion_log.append(float(qpos[socket_flexion_qpos_adr]))
        socket_flexion_qvel_log.append(float(qvel[socket_flexion_qvel_adr]))    # NEW

        if pylon_torque_adr is not None:
            pylon_moment_log.append(float(sensordata[pylon_torque_adr + 2]))
        else:
            pylon_moment_log.append(np.nan)

        # corrected ankle moment sensor, if present in the model
        if prosthetic_ankle_torque_adr is not None:
            prosthetic_ankle_moment_log.append(float(sensordata[prosthetic_ankle_torque_adr + 2]))
        else:
            prosthetic_ankle_moment_log.append(np.nan)

    # Ground reaction force
    grf_left = collect_grf_vertical(env_state.data, grf_body_ids["left"])
    grf_right = collect_grf_vertical(env_state.data, grf_body_ids["right"])
    grf_left_log.append(grf_left)
    grf_right_log.append(grf_right)

    # Pelvis / foot positions
    pelvis_pos_log.append(env_state.data.xpos[0][position_body_ids["pelvis"]].copy())
    foot_left_pos_log.append(env_state.data.xpos[0][position_body_ids["foot_left"]].copy())
    foot_right_pos_log.append(env_state.data.xpos[0][position_body_ids["foot_right"]].copy())

    if args.record:
        env.mjx_render(env_state, record=True)

    if i % 100 == 0:
        if is_esr:
            print(f"Step {i:5d}: ESR theta = {np.rad2deg(esr_hinge_log['angle'][-1]):8.3f} deg")
        else:
            print(f"Step {i:5d}: (no ESR hinge)")

env.stop()


# =============================================================================
# Convert recorded data to NumPy arrays
steps = np.asarray(steps)
sim_time = np.asarray(sim_time_log[:-1])

angle_deg = {}
velocity = {}
moment = {}
power = {}
ref_angle_deg = {}

for key in joint_info:
    angle_deg[key] = np.rad2deg(np.asarray(logs[key]["angle"]))
    velocity[key] = np.asarray(logs[key]["vel"])
    moment[key] = np.asarray(logs[key]["moment"])
    power[key] = moment[key] * velocity[key]
    ref_angle_deg[key] = np.rad2deg(np.asarray(ref_logs[key]))

right_foot_mimic_moment = np.asarray(right_foot_mimic_moment_log)

# Unify "ankle_right" across ESR / SACH (unchanged, kept for backward compatibility)
if is_esr:
    esr_angle_deg = np.rad2deg(np.asarray(esr_hinge_log["angle"]))
    esr_velocity  = np.asarray(esr_hinge_log["vel"])
    esr_hinge_moment = np.asarray(esr_hinge_log["moment"]) # internal hinge moment, diagnostic only
    esr_hinge_power = esr_hinge_moment * esr_velocity

    angle_deg["ankle_right"] = esr_angle_deg
    velocity["ankle_right"]  = esr_velocity
    moment["ankle_right"]    = right_foot_mimic_moment # consistent sensor source across subtypes
    power["ankle_right"] = moment["ankle_right"] * velocity["ankle_right"]
    # No ESR degrees of freedom exist in the reference trajectory
    # ref_angle_deg["ankle_right"] = np.full_like(steps, np.nan, dtype=float)

    # Segment-based (shank/pylon-to-forefoot) ankle angle: hinge rotation PLUS the static socket alignment offset. This is the closer analogue to the clinical "ankle angle" definition (shank segment vs. foot segment) than esr_hinge_r qpos alone. 
    socket_flexion_deg = np.rad2deg(np.asarray(socket_flexion_log))
    ankle_right_segment_angle_deg = esr_angle_deg + socket_flexion_deg

    # Total moment transmitted through the pylon into the socket/residual limb. 
    # Power here is only an APPROXIMATION: it pairs the pylon moment with the hinge's angular velocity, since the hinge is the only rotational DOF available between talus_r and esr_forefoot_r in this model.
    esr_pylon_moment = np.asarray(pylon_moment_log)
    esr_pylon_power = esr_pylon_moment * esr_velocity

    # =========================================================================
    # applied-torque reconstruction (tau_ESR, tau_mujoco_spring, tau_applied)
    esr_theta_rad = np.asarray(esr_hinge_log["angle"])  # radians, as logged
    esr_tau_ESR, esr_tau_mujoco_spring, esr_tau_applied = compute_esr_diagnostic_torques(
        esr_theta_rad, ESR_MODEL_TYPE_RESOLVED, ESR_LEVER_ARM, ESR_HINGE_BASE_STIFFNESS,
        LINEAR_PARAMS, NONLINEAR_PARAMS)

    # =========================================================================
    # corrected ankle_right angle / moment / power for the prosthesis side
    socket_flexion_qvel_arr = np.asarray(socket_flexion_qvel_log)
    segment_angular_velocity = esr_velocity + socket_flexion_qvel_arr  # rad/s 
    ankle_right_angle_deg_lit = -ankle_right_segment_angle_deg

    prosthetic_ankle_moment = np.asarray(prosthetic_ankle_moment_log)
    used_prosthetic_ankle_sensor = not np.all(np.isnan(prosthetic_ankle_moment))
    if not used_prosthetic_ankle_sensor:
        print("\nWARNING: 'prosthetic_ankle_r_torque_sensor' was not found in this "
              "model -- ankle_right_moment_corrected falls back to "
              "right_foot_mimic_torque_sensor (calcn_r location, i.e. NOT the "
              "pylon-rearfoot socket interface). This is a less accurate proxy "
              "for the literature ankle moment. Add the new site+sensor to "
              "MjxSkeletonMuscleProsthesis and re-run for the corrected value.\n")
        ankle_right_moment_corrected = right_foot_mimic_moment.copy()
    else:
        ankle_right_moment_corrected = prosthetic_ankle_moment
    ankle_right_power_corrected = ankle_right_moment_corrected * segment_angular_velocity

else:
    moment["ankle_right"] = right_foot_mimic_moment
    power["ankle_right"]  = moment["ankle_right"] * velocity["ankle_right"]
    # direct ESR measured values
    esr_hinge_moment = np.full_like(steps, np.nan, dtype=float)
    esr_hinge_power  = np.full_like(steps, np.nan, dtype=float)
    ankle_right_segment_angle_deg = np.full_like(steps, np.nan, dtype=float)
    # Pylon measurements for ESR
    esr_pylon_moment = np.full_like(steps, np.nan, dtype=float)
    esr_pylon_power  = np.full_like(steps, np.nan, dtype=float)
    # tau applied
    esr_tau_ESR           = np.full_like(steps, np.nan, dtype=float)
    esr_tau_mujoco_spring = np.full_like(steps, np.nan, dtype=float)
    esr_tau_applied       = np.full_like(steps, np.nan, dtype=float)
    # For SACH, the "corrected" ankle quantities are just the real joint's values
    ankle_right_angle_deg_lit    = angle_deg["ankle_right"]
    ankle_right_moment_corrected = moment["ankle_right"]
    ankle_right_power_corrected  = power["ankle_right"]

# Ground reaction force
grf_left  = np.asarray(grf_left_log)
grf_right = np.asarray(grf_right_log)


pelvis_pos     = np.asarray(pelvis_pos_log)
foot_left_pos  = np.asarray(foot_left_pos_log)
foot_right_pos = np.asarray(foot_right_pos_log)
print("Pelvis position range per axis (x,y,z):", pelvis_pos.max(axis=0) - pelvis_pos.min(axis=0))


# =============================================================================
# ESR ROM analysis
esr_rom = None
lecomte_min, lecomte_max = -6.0, 14.0
if is_esr:
    esr_theta_min = angle_deg["ankle_right"].min()
    esr_theta_max = angle_deg["ankle_right"].max()
    esr_rom = esr_theta_max - esr_theta_min

    print("\n" + "=" * 60)
    print("ESR ROM ANALYSIS")
    print("=" * 60)
    print(f"Theta min: {esr_theta_min:.3f} deg")
    print(f"Theta max: {esr_theta_max:.3f} deg")
    print(f"ROM:       {esr_rom:.3f} deg")

    print("\n\nLecomte reference:")
    print(f"  Minimum: {lecomte_min:.1f} deg")
    print(f"  Maximum: {lecomte_max:.1f} deg")
    print(f"  ROM:     {lecomte_max - lecomte_min:.1f} deg")

# =============================================================================
# GRF sanity check
print("\n" + "=" * 60)
print("GRF SANITY CHECK")
print("=" * 60)
if args.body_mass_kg is not None:
    expected_weight_n = args.body_mass_kg * 9.81
    mean_total_vertical = float(grf_left.mean() + grf_right.mean())
    ratio = mean_total_vertical / expected_weight_n
    print(f"Mean(left) + Mean(right) vertical GRF: {mean_total_vertical:.1f} N")
    print(f"Expected body weight:                  {expected_weight_n:.1f} N")
    print(f"Ratio:                                 {ratio:.2f}")
    print("A ratio far from ~1.0 suggests the signal is contaminated by "
          "mimic/tracking constraint forces rather than pure ground contact")
else:
    print("Pass --body_mass_kg to enable this check "
          "(compares mean vertical GRF to expected body weight).")


# =============================================================================
# Plotting functions
def get_plot_dir(prosthesis_subtype_, esr_model_type_=None, run_timestamp_=None):
    """
    Build and create a plot directory based on prosthesis subtype, optional ESR model type, and optional run timestamp.
    """
    base_dir = "plots"
    plot_dir = os.path.join(base_dir, prosthesis_subtype_)
    if prosthesis_subtype_ == "ESR" and esr_model_type_ is not None:
        plot_dir = os.path.join(plot_dir, esr_model_type_)
    if run_timestamp_ is not None:
        plot_dir = os.path.join(plot_dir, run_timestamp_)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def plot_function(steps_, curves, title, ylabel, filename, prosthesis_subtype_,
                   esr_model_type_=None, run_timestamp_=None, hlines=None, fill=None):
    """
    Plots curves, optional hlines and fill, applies styling, saves PNG, and shows the figure.
    curves: list of (y_values, label)
    hlines: list of (y, linestyle, label), optional
    fill: (y_min, y_max), optional
    """
    plt.figure(figsize=(12, 6))
    for y, label in curves:
        plt.plot(steps_, y, linewidth=0.8, label=label)
    if hlines:
        for y, style, label in hlines:
            plt.axhline(y, linestyle=style, alpha=0.5, linewidth=1.5, label=label)
    if fill:
        y1, y2 = fill
        plt.fill_between(steps_, y1, y2, alpha=0.05)
    plt.xlabel("Evaluation step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_dir = get_plot_dir(prosthesis_subtype_, esr_model_type_, run_timestamp_)
    full_path = os.path.join(plot_dir, filename)
    plt.savefig(full_path, dpi=150)
    print(f"Saved {full_path}")
    #plt.show()


# ESR hinge angle plot (ESR only)
if is_esr:
    plot_function(
        steps,
        curves=[(angle_deg["ankle_right"], "ESR hinge angle"),
                (ankle_right_segment_angle_deg, "Segment angle (hinge + alignment offset)")],
        title=f"ESR Hinge Angle — Observed ROM = {esr_rom:.2f}°",
        ylabel=r"$\theta$ [deg]",
        filename="esr_rom_analysis.png",
        prosthesis_subtype_=prosthesis_subtype,
        esr_model_type_=esr_model_type,
        run_timestamp_=run_timestamp,
        hlines=[(0, "--", None), (lecomte_max, ":", "Lecomte +14°"), (lecomte_min, ":", "Lecomte -6°")],
        fill=(lecomte_min, lecomte_max))


    # applied-torque, validates the qfrc_applied injection
    plot_function(
        steps,
        curves=[(esr_tau_ESR, "tau_ESR (desired, Lecomte law)"),
                (esr_hinge_moment, "Actual sensed joint torque"),
                (esr_tau_applied, "tau_applied (qfrc_applied correction)"),
                (esr_tau_mujoco_spring, "tau_mujoco_spring (native spring)"),
                (ankle_right_moment_corrected, "prosthetic_ankle_moment (socket reaction)")],
        title="ESR Applied Torque",
        ylabel="Moment [Nm]",
        filename="esr_applied_torque.png",
        prosthesis_subtype_=prosthesis_subtype,
        esr_model_type_=esr_model_type,
        run_timestamp_=run_timestamp,
        hlines=[(0, "--", None)])

    # corrected ankle_right angle vs. the original (uncorrected) versions, for a direct before/after comparison.
    plot_function(
        steps,
        curves=[(-ankle_right_segment_angle_deg, "Corrected (segment, lit. sign)"),
                (angle_deg["ankle_right"], "Original (raw hinge)")],
        title="Ankle Angle (Prosthesis side): Corrected vs. Original",
        ylabel="Angle [deg] (Dorsi = +)",
        filename="ankle_right_angle_corrected_vs_original.png",
        prosthesis_subtype_=prosthesis_subtype,
        esr_model_type_=esr_model_type,
        run_timestamp_=run_timestamp,
        hlines=[(0, "--", None)])


# ======= Basic Comparison Plots =======
# Generic angle / moment / power plots per joint pair
JOINT_PAIRS = [
    ("ankle", "Ankle", "deg", "Nm", "W"),
    ("knee", "Knee", "deg", "Nm", "W"),
    ("hip", "Hip", "deg", "Nm", "W")]

# Plot all angle, moment and power plots
for prefix, label, angle_unit, moment_unit, power_unit in JOINT_PAIRS:
    left_key, right_key = f"{prefix}_left", f"{prefix}_right"

    plot_function(
        steps,
        curves=[(angle_deg[left_key], "Left / intact"), (angle_deg[right_key], "Right / prosthesis")],
        title=f"{label} Joint Angle",
        ylabel=f"{label} angle [{angle_unit}]",
        filename=f"{prefix}_angle.png",
        prosthesis_subtype_=prosthesis_subtype,
        esr_model_type_=esr_model_type,
        run_timestamp_=run_timestamp)

    plot_function(
        steps,
        curves=[(moment[left_key], "Left / intact"), (moment[right_key], "Right / prosthesis")],
        title=f"{label} Joint Moment",
        ylabel=f"{label} moment [{moment_unit}]",
        filename=f"{prefix}_moment.png",
        prosthesis_subtype_=prosthesis_subtype,
        esr_model_type_=esr_model_type,
        run_timestamp_=run_timestamp)

    plot_function(
        steps,
        curves=[(power[left_key], "Left / intact"), (power[right_key], "Right / prosthesis")],
        title=f"{label} Joint Power",
        ylabel=f"{label} power [{power_unit}]",
        filename=f"{prefix}_power.png",
        prosthesis_subtype_=prosthesis_subtype,
        esr_model_type_=esr_model_type,
        run_timestamp_=run_timestamp,
        hlines=[(0, "--", None)])

# GRF plot
plot_function(
    steps,
    curves=[(grf_left, "Left / intact"), (grf_right, "Right / prosthesis")],
    title="Vertical Ground Reaction Force (toes + calcn)",
    ylabel="Vertical GRF [N]",
    filename="grf_vertical.png",
    prosthesis_subtype_=prosthesis_subtype,
    esr_model_type_=esr_model_type,
    run_timestamp_=run_timestamp,
    hlines=[(0, "--", None)])

# =============================================================================
# CSV export -- identical column set for both ESR and SACH runs
def export_raw_data_to_csv(angle_deg_, moment_, power_, ref_angle_deg_,
                            esr_hinge_moment_, esr_hinge_power_,
                            ankle_right_segment_angle_deg_, esr_pylon_moment_, esr_pylon_power_,
                            esr_tau_ESR_, esr_tau_mujoco_spring_, esr_tau_applied_,
                            ankle_right_angle_deg_lit_,
                            ankle_right_moment_corrected_, ankle_right_power_corrected_,
                            grf_left_, grf_right_, pelvis_pos_, foot_left_pos_, foot_right_pos_,
                            steps_, sim_time_,
                            prosthesis_subtype_, esr_model_type_, run_timestamp_,
                            filename="raw_joint_data.csv"):
    """
    Save all raw data to a CSV file. Columns are generated generically from the joint dicts, so the same column names appear for ESR and SACH runs (ankle_right_* included in both cases).
    """
    data = {"step": steps_, "sim_time": sim_time_}

    for key in angle_deg_:
        data[f"{key}_angle_deg"] = angle_deg_[key]
        data[f"{key}_moment"] = moment_[key]
        data[f"{key}_power"] = power_[key]
        data[f"ref_{key}_angle_deg"] = ref_angle_deg_[key]

    # ESR-only diagnostic columns (NaN for SACH runs)
    data["esr_hinge_moment"] = esr_hinge_moment_
    data["esr_hinge_power"]  = esr_hinge_power_
    # Segment-based (shank/pylon-to-forefoot) ankle angle: hinge + static socket alignment offset.
    data["ankle_right_segment_angle_deg"] = ankle_right_segment_angle_deg_
    # Total moment transmitted through the pylon into the socket/residual limb, plus an approximate power
    data["esr_pylon_moment"] = esr_pylon_moment_
    data["esr_pylon_power"] = esr_pylon_power_

    # applied-torque reconstruction
    data["esr_tau_ESR"] = esr_tau_ESR_
    data["esr_tau_mujoco_spring"] = esr_tau_mujoco_spring_
    data["esr_tau_applied"] = esr_tau_applied_

    # moment: prosthetic_ankle sensor if available, else right_foot_mimic fallback)
    data["ankle_right_angle_deg_lit_convention"] = ankle_right_angle_deg_lit_
    data["ankle_right_moment_corrected"] = ankle_right_moment_corrected_
    data["ankle_right_power_corrected"]  = ankle_right_power_corrected_

    data["grf_left"] = grf_left_
    data["grf_right"] = grf_right_

    data["pelvis_x"], data["pelvis_y"], data["pelvis_z"] = (
        pelvis_pos_[:, 0], pelvis_pos_[:, 1], pelvis_pos_[:, 2])
    data["foot_left_x"], data["foot_left_y"], data["foot_left_z"] = (
        foot_left_pos_[:, 0], foot_left_pos_[:, 1], foot_left_pos_[:, 2])
    data["foot_right_x"], data["foot_right_y"], data["foot_right_z"] = (
        foot_right_pos_[:, 0], foot_right_pos_[:, 1], foot_right_pos_[:, 2])

    df = pd.DataFrame(data)
    plot_dir = get_plot_dir(prosthesis_subtype_, esr_model_type_, run_timestamp_)
    full_path = os.path.join(plot_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved raw data to {full_path}")


export_raw_data_to_csv(
    angle_deg, moment, power, ref_angle_deg,
    esr_hinge_moment, esr_hinge_power,
    ankle_right_segment_angle_deg, esr_pylon_moment, esr_pylon_power,
    esr_tau_ESR, esr_tau_mujoco_spring, esr_tau_applied,
    ankle_right_angle_deg_lit,
    ankle_right_moment_corrected, ankle_right_power_corrected,
    grf_left, grf_right, pelvis_pos, foot_left_pos, foot_right_pos,
    steps, sim_time,
    prosthesis_subtype, esr_model_type, run_timestamp,
    filename="raw_joint_data.csv")


# =============================================================================
# Body listing + prosthesis axis identification (diagnostic, kept from original)
# for i in range(mj_model.nbody):
#     print(i, mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i))


# def identify_vertical_axis(model, data, body_name, world_up=(0.0, 0.0, 1.0)):
#     """
#     Determine which LOCAL axis (x/y/z) of a given body is most aligned
#     with the world vertical ("up") direction at the reference pose.
#     """
#     body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
#     if body_id < 0:
#         raise RuntimeError(f"Could not find body '{body_name}' in the model.")

#     R = np.asarray(data.xmat[0, body_id])
#     assert R.shape == (3, 3), f"xmat for body {body_name} has wrong shape: {R.shape}"

#     world_up_vec = np.asarray(world_up, dtype=float)
#     world_up_vec /= np.linalg.norm(world_up_vec)

#     dots = {}
#     for i, axis_name in enumerate(["x", "y", "z"]):
#         local_axis_world = R[:, i]
#         dots[axis_name] = float(np.dot(local_axis_world, world_up_vec))
#         print(f"  local {axis_name}-axis in world frame: {np.round(local_axis_world, 3)}  "
#               f"(dot with world up: {dots[axis_name]:+.3f})")

#     best_axis = max(dots, key=lambda k: abs(dots[k]))
#     print(f"-> '{best_axis}' is most aligned with world vertical "
#           f"(|dot|={abs(dots[best_axis]):.3f})")
#     return best_axis, dots


# _ref_data = env_state.data
# print("\nIdentifying pistoning (proximal-distal) axis for pylon_socket:")
# identify_vertical_axis(mj_model, _ref_data, "pylon_socket_r")