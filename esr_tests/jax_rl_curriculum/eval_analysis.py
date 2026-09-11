"""
This script evaluates a trained PPOJax locomotion policy using the MJX
simulation and analyses the motion and joint dynamics of the prosthetic
and intact legs.

During the evaluation, the following quantities are recorded externally
from the MJX simulation state:
    - ESR hinge angle
    - Hip angles (left and right)
    - Knee angles (left and right)
    - Hip moments (left and right)
    - Knee moments (left and right)
    - Vertical ground reaction force (left and right), summed over the
      toes + calcaneus bodies

Joint powers are calculated afterwards from the corresponding joint moment
and angular velocity:
    P = M * q_dot

The observed ESR range of motion (ROM) is calculated and compared with
the reference ROM reported by Lecomte et al.

The script uses the MJX evaluation path and must therefore be run without
the --use_mujoco option.

The --record option can optionally be used to record the MJX rendering.

Example:
    python esr_eval.py --path path/to/PPOJax_saved.pkl --n_steps 1000 --record --body_mass_kg 86.6
"""

import os

# Uncomment the following lines to force JAX to use the CPU instead of GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax

# Uncomment the following line to force JAX to use the CPU instead of GPU.
# jax.config.update("jax_platform_name", "cpu")

import argparse

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf


os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

# headless
os.environ["MUJOCO_GL"] = "egl"


# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate PPOJax agent and analyse prosthesis gait (ESR or SACH).")
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--n_steps', type=int, default=1000, help='Number of evaluation steps')
parser.add_argument('--record', action='store_true', help='Record MJX rendering')
parser.add_argument('--body_mass_kg', type=float, default=None,
                     help='Body mass in kg. Optional -- only used for a sanity check '
                          '(mean vertical GRF across both legs vs. body weight).')
args = parser.parse_args()

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
        "pylon_socket": {
            "x": [-0.0, 0.0],
            "z": [-0.0, 0.0]
        },
        "talus": {
            "x": [-0.0, 0.0],
            "z": [-0.0, 0.0]
        }
    },
    "randomize_prosthesis_body_orientation": True,
    "prosthesis_body_orientation_range": {
        "pylon_socket": {
            "x": [-0.0, 0.0],
            "y": [-0.0, 0.0],
            "z": [-0.0, 0.0]
        },
        "talus": {
            "x": [-0.0, 0.0],
            "z": [-0.0, 0.0]
        }
    }
}

# create env
OmegaConf.set_struct(config, False) # Allow modifications
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
esr_model_type = config.experiment.env_params.get("ESR_model_type", None)
prosthesis_subtype = config.experiment.env_params["prosthesis_subtype"]
is_esr = (prosthesis_subtype == "ESR")
is_sach = (prosthesis_subtype == "SACH")


# Find underlying MuJoCo model
def find_mujoco_model(obj):
    """
    Recursively search through the environment wrappers for the underlying
    MuJoCo model.
    The exact location of the MuJoCo model depends on the environment and
    wrapper structure. This helper therefore checks several common attribute
    names and recursively searches through nested environment wrappers.
    """
    visited = set()

    def search(x):
        if x is None:
            return None
        # Prevent infinite recursion if the same object is encountered again.
        if id(x) in visited:
            return None
        visited.add(id(x))

        # Check for direct MuJoCo model attributes.
        for name in ["mj_model", "model", "_model"]:
            if hasattr(x, name):
                candidate = getattr(x, name)
                if candidate is None:
                    continue
                # Basic check that the candidate is a MuJoCo model.
                if (hasattr(candidate, "jnt_qposadr") and hasattr(candidate, "joint")):
                    return candidate

        # Search through environment wrappers.
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

# print("\nJoints in MuJoCo model:")
# for i in range(mj_model.njnt):
#     joint_name = mujoco.mj_id2name(
#         mj_model,
#         mujoco.mjtObj.mjOBJ_JOINT,
#         i
#     )
#     qpos_adr = int(mj_model.jnt_qposadr[i])
#     qvel_adr = int(mj_model.jnt_dofadr[i])

#     print(
#         f"  ID {i:2d}: {joint_name:30s} "
#         f"qpos={qpos_adr:2d}, qvel={qvel_adr:2d}"
#     )

# print("\nSensors in MuJoCo model:")

# for i in range(mj_model.nsensor):
#     sensor_name = mujoco.mj_id2name(
#         mj_model,
#         mujoco.mjtObj.mjOBJ_SENSOR,
#         i
#     )

#     print(
#         f"  ID {i:2d}: {sensor_name:40s} "
#         f"adr={int(mj_model.sensor_adr[i])}, "
#         f"dim={int(mj_model.sensor_dim[i])}"
#     )


# print("\nBodies in MuJoCo model:")
# for i in range(mj_model.nbody):
#     body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
#     print(f"  ID {i:3d}: {body_name}")

# =============================================================================
# Helper functions for joints and sensors
def get_joint_qpos_address(model, joint_name):
    """
    Return the qpos address of a named MuJoCo joint.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Could not find joint '{joint_name}' in the MuJoCo model.")
    return int(model.jnt_qposadr[joint_id])


def get_joint_qvel_address(model, joint_name):
    """
    Return the qvel address of a named MuJoCo joint.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Could not find joint '{joint_name}' in the MuJoCo model.")
    return int(model.jnt_dofadr[joint_id])


def get_sensor_id(model, sensor_name):
    """
    Return the MuJoCo sensor ID for a named sensor.
    """
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise RuntimeError(f"Could not find sensor '{sensor_name}' in the MuJoCo model.")
    return int(sensor_id)


def get_sensor_adr(model, sensor_id):
    """
    Return the starting address of a sensor in sensordata.
    """
    return int(model.sensor_adr[sensor_id])


def get_body_id(model, body_name):
    """
    Return the MuJoCo body ID for a named body.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body '{body_name}' in the MuJoCo model.")
    return int(body_id)


# Find ESR hinge joint
if is_esr:
    HINGE_NAME = "esr_hinge_r"  # Name of the ESR hinge joint in the MuJoCo model
    # Obtain the numerical MuJoCo joint ID from its name.
    hinge_qpos_adr = get_joint_qpos_address(mj_model, HINGE_NAME)
    hinge_qvel_adr = get_joint_qvel_address(mj_model, HINGE_NAME)
    # print(f"ESR hinge qpos address: {hinge_qpos_adr}")
    # print(f"ESR hinge qvel address: {hinge_qvel_adr}")
else: 
    hinge_qpos_adr = None
    hinge_qvel_adr = None

# -------------------------------------------------------------------------
# Hip and knee joints
JOINT_NAMES = {
    "hip_left": "hip_flexion_l",
    "hip_right": "hip_flexion_r",
    "knee_left": "knee_angle_l",
    "knee_right": "knee_angle_r",
}
joint_qpos_adr = {}
joint_qvel_adr = {}
for name, joint_name in JOINT_NAMES.items():
    joint_qpos_adr[name] = get_joint_qpos_address(mj_model, joint_name)
    joint_qvel_adr[name] = get_joint_qvel_address(mj_model, joint_name)
    # print(f"{name}: "
    #       f"qpos={joint_qpos_adr[name]}, "
    #       f"qvel={joint_qvel_adr[name]}")
# -------------------------------------------------------------------------
# Identify torque sensors
TORQUE_SENSOR_NAMES = {
    "hip_left": "left_hip_mimic_torque_sensor",
    "hip_right": "right_hip_mimic_torque_sensor",
    "knee_left": "left_knee_mimic_torque_sensor",
    "knee_right": "right_knee_mimic_torque_sensor",}
torque_sensor_adr = {}
for name, sensor_name in TORQUE_SENSOR_NAMES.items():
    sensor_id = get_sensor_id(mj_model, sensor_name)
    torque_sensor_adr[name] = get_sensor_adr(mj_model, sensor_id)
    # print(f"{name} torque sensor: "
    #       f"ID={sensor_id}, "
    #       f"address={torque_sensor_adr[name]}")

# -------------------------------------------------------------------------
# Ground reaction force bodies (toes + calcaneus per side). We sum both
# since the foot contact is typically split across a heel (calcn) and a
# forefoot (toes) geom/body.
if is_esr:
    right_bodies = ["talus_r", "esr_forefoot_r"]
else:
    right_bodies = ["toes_r", "calcn_r"]

GRF_BODY_NAMES = {
    "left": ["toes_l", "calcn_l"],
    "right": right_bodies,
}


VERTICAL_FORCE_INDEX = 5

grf_body_ids = {
    side: [get_body_id(mj_model, name) for name in names]
    for side, names in GRF_BODY_NAMES.items()
}


def collect_grf_vertical(data, body_ids, vertical_index=VERTICAL_FORCE_INDEX, env_idx=0):
    """
    Sum the vertical component of cfrc_ext (external force on each body,
    world frame) across the given body ids, for a single env index.
    """
    total = 0.0
    for body_id in body_ids:
        total += float(data.cfrc_ext[env_idx, body_id, vertical_index])
    return total


# -------------------------------------------------------------------------
# Use the same wrapping as PPOJax.play_policy()
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
    Compute an action from the trained PPO policy.

    The running observation statistics are updated in the same way as
    in PPOJax.play_policy().
    """
    y, updates = agent_conf.network.apply(
        {"params": ts.params, "run_stats": ts.run_stats},
        obs,
        mutable=["run_stats"])
    ts = ts.replace(run_stats=updates["run_stats"])
    pi, _ = y
    action = pi.sample(seed=_rng)

    return action, ts

# JIT compile
print("\nJIT compiling policy...")
plcy_call = jax.jit(sample_actions)

# Reset
obs, env_state = env.reset(env_keys)

# Initialize evaluation data
steps = []
# ESR
esr_theta_log = []
# Hip angles
hip_angle_left_log = []
hip_angle_right_log = []
# Knee angles
knee_angle_left_log = []
knee_angle_right_log = []
# Hip moments
hip_moment_left_log = []
hip_moment_right_log = []
# Knee moments
knee_moment_left_log = []
knee_moment_right_log = []
# Hip angular velocities
hip_velocity_left_log = []
hip_velocity_right_log = []
# Knee angular velocities
knee_velocity_left_log = []
knee_velocity_right_log = []
# Vertical ground reaction force (toes + calcn per side)
grf_left_log = []
grf_right_log = []


# ============================================================
# Evaluation
print("\nStarting evaluation...")

for i in range(args.n_steps):

    # Random key
    rng, _rng = jax.random.split(rng)

    # Policy action
    action, train_state = plcy_call(train_state, obs, _rng)
    action = jnp.atleast_2d(action)

    # MJX step
    (obs, reward, absorbing, done, info, env_state) = env.step(env_state,action)

    # Read qpos and qvel from MJX state
    qpos = env_state.data.qpos[0]
    qvel = env_state.data.qvel[0]

    if is_esr:
        # ESR angle
        esr_theta = float(qpos[hinge_qpos_adr])
        esr_theta_log.append(esr_theta)
    else:
        esr_theta = None

    # Hip angles
    hip_left  = float(qpos[joint_qpos_adr["hip_left"]])
    hip_right = float(qpos[joint_qpos_adr["hip_right"]])
    # Knee angles
    knee_left  = float(qpos[joint_qpos_adr["knee_left"]])
    knee_right = float(qpos[joint_qpos_adr["knee_right"]])

    # Angular velocities
    hip_velocity_left   = float(qvel[joint_qvel_adr["hip_left"]])
    hip_velocity_right  = float(qvel[joint_qvel_adr["hip_right"]])
    knee_velocity_left  = float(qvel[joint_qvel_adr["knee_left"]])
    knee_velocity_right = float(qvel[joint_qvel_adr["knee_right"]])

    # Read torque sensors (here z-component (+2) is choen as sagittal-plane moment)
    sensordata = env_state.data.sensordata[0]
    hip_moment_left   = float(sensordata[torque_sensor_adr["hip_left"] + 2]) 
    hip_moment_right  = float(sensordata[torque_sensor_adr["hip_right"] + 2])
    knee_moment_left  = float(sensordata[torque_sensor_adr["knee_left"] + 2])
    knee_moment_right = float(sensordata[torque_sensor_adr["knee_right"] + 2])

    # Vertical ground reaction force (toes + calcn), for downstream
    # heel-strike / gait-event detection.
    grf_left  = collect_grf_vertical(env_state.data, grf_body_ids["left"])
    grf_right = collect_grf_vertical(env_state.data, grf_body_ids["right"])

    # Store data
    steps.append(i)
    hip_angle_left_log.append(hip_left)
    hip_angle_right_log.append(hip_right)
    knee_angle_left_log.append(knee_left)
    knee_angle_right_log.append(knee_right)
    hip_velocity_left_log.append(hip_velocity_left)
    hip_velocity_right_log.append(hip_velocity_right)
    knee_velocity_left_log.append(knee_velocity_left)
    knee_velocity_right_log.append(knee_velocity_right)
    hip_moment_left_log.append(hip_moment_left)
    hip_moment_right_log.append(hip_moment_right)
    knee_moment_left_log.append(knee_moment_left)
    knee_moment_right_log.append(knee_moment_right)
    grf_left_log.append(grf_left)
    grf_right_log.append(grf_right)

    # MJX rendering
    if args.record:
        env.mjx_render(env_state, record=True)

    # Progress output
    if i % 100 == 0:
        if is_esr:
            print(f"Step {i:5d}: "
            f"ESR theta = {np.rad2deg(esr_theta):8.3f} deg")
        else:
            print(f"Step {i:5d}: (no ESR hinge)")

env.stop()

# Convert recorded data to NumPy arrays
steps = np.asarray(steps)

# ESR
esr_theta = np.asarray(esr_theta_log)

# Angles
hip_angle_left   = np.asarray(hip_angle_left_log)
hip_angle_right  = np.asarray(hip_angle_right_log)
knee_angle_left  = np.asarray(knee_angle_left_log)
knee_angle_right = np.asarray(knee_angle_right_log)

# Angular velocities
hip_velocity_left   = np.asarray(hip_velocity_left_log)
hip_velocity_right  = np.asarray(hip_velocity_right_log)
knee_velocity_left  = np.asarray(knee_velocity_left_log)
knee_velocity_right = np.asarray(knee_velocity_right_log)

# Moments
hip_moment_left   = np.asarray(hip_moment_left_log)
hip_moment_right  = np.asarray(hip_moment_right_log)
knee_moment_left  = np.asarray(knee_moment_left_log)
knee_moment_right = np.asarray(knee_moment_right_log)

# Ground reaction force
grf_left  = np.asarray(grf_left_log)
grf_right = np.asarray(grf_right_log)

# Sanity check: a GRF channel that is (near) constantly zero almost always
# means the configured bodies for that side carry no contact geometry
# (e.g. wrong body names -- see the body listing printed near the start of
# this script), not that the leg never touches the ground.
for side_name, grf_arr in [("left", grf_left), ("right", grf_right)]:
    if np.max(np.abs(grf_arr)) < 1.0:
        print(f"WARNING: grf_{side_name} is ~0 N for the entire run. "
              f"This almost certainly means GRF_BODY_NAMES['{side_name}'] "
              f"points to bodies with no contact geometry -- check the "
              f"body listing above and update GRF_BODY_NAMES.")

# Convert angles from radians to degrees.
esr_theta_deg        = np.rad2deg(esr_theta)
hip_angle_left_deg   = np.rad2deg(hip_angle_left)
hip_angle_right_deg  = np.rad2deg(hip_angle_right)
knee_angle_left_deg  = np.rad2deg(knee_angle_left)
knee_angle_right_deg = np.rad2deg(knee_angle_right)

# ------------------------------------------------------------
# Joint power: P = M * q_dot  [Nm * rad/s = W]
hip_power_left   = (hip_moment_left * hip_velocity_left)
hip_power_right  = (hip_moment_right * hip_velocity_right)
knee_power_left  = (knee_moment_left * knee_velocity_left)
knee_power_right = (knee_moment_right * knee_velocity_right)

# ESR ROM
if is_esr:
    esr_theta_min = esr_theta_deg.min()
    esr_theta_max = esr_theta_deg.max()
    esr_rom = esr_theta_max - esr_theta_min
    print("ESR ROM ANALYSIS")
    print("\n")
    print("=" * 60)
    print("ESR ROM ANALYSIS")
    print("=" * 60)

    print(f"Theta min: {esr_theta_min:.3f} deg")
    print(f"Theta max: {esr_theta_max:.3f} deg")
    print(f"ROM:       {esr_rom:.3f} deg")
    print("=" * 60)
    # Lecomte reference
    lecomte_min = -6.0
    lecomte_max = 14.0
    lecomte_rom = (lecomte_max - lecomte_min)

    print("\nLecomte reference:")
    print(f"  Minimum: {lecomte_min:.1f} deg")
    print(f"  Maximum: {lecomte_max:.1f} deg")
    print(f"  ROM:     {lecomte_rom:.1f} deg")
else:
    esr_rom = None


# ------------------------------------------------------------
# GRF sanity check: mean(vertical GRF, both legs) over the recorded steps
# should be close to body weight (mean vertical CoM acceleration ~0 over a
# long-enough steady-state walk). Only printed if --body_mass_kg was given.
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
          "mimic/tracking constraint forces rather than pure ground "
          "contact")
else:
    print("Pass --body_mass_kg to enable this check "
          "(compares mean vertical GRF to expected body weight).")


# ============================================================
# Plot
import matplotlib.pyplot as plt

def get_plot_dir(prosthesis_subtype, esr_model_type=None):
    base_dir = "plots"

    # First level: ESR or SACH
    plot_dir = os.path.join(base_dir, prosthesis_subtype)

    # Second level: linear / nonlinear only for ESR
    if prosthesis_subtype == "ESR" and esr_model_type is not None:
        plot_dir = os.path.join(plot_dir, esr_model_type)

    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir



def plot_function(steps, curves, title, ylabel, filename, prosthesis_subtype, esr_model_type=None, hlines=None, fill=None):
    """
    Plots curves, optional hlines and fill, applies styling, saves PNG, and shows the figure.

    steps : array-like                              -> X-axis values.
    curves : list of (array-like, str)              -> Curves to plot, each as (y_values, label).
    title : str                                     -> Plot title.
    ylabel : str                                    -> Y-axis label.
    filename : str                                  -> Output PNG filename.
    hlines : list of (float, str, str), optional    -> Horizontal lines as (y, linestyle, label).
    fill : (float, float), optional                 -> Shaded region between (y_min, y_max).
    """
    plt.figure(figsize=(12, 6))
    # plot each curve
    for y, label in curves:
        plt.plot(steps, y, linewidth=0.8, label=label)
    # plot horizontal lines if provided
    if hlines:
        for y, style, label in hlines:
            plt.axhline(y, linestyle=style, alpha=0.5, linewidth=1.5, label=label)
    # fill between if provided
    if fill:
        y1, y2 = fill
        plt.fill_between(steps, y1, y2, alpha=0.05)

    plt.xlabel("Evaluation step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_dir = get_plot_dir(prosthesis_subtype, esr_model_type)
    full_path = os.path.join(plot_dir, filename)
    plt.savefig(full_path, dpi=150)
    print(f"Saved {full_path}")
    plt.show()

# Plot ESR hinge angle
if is_esr:
    plot_function(
        steps,
        curves=[(esr_theta_deg, "ESR hinge angle")],
        title=f"ESR Hinge Angle — Observed ROM = {esr_rom:.2f}°",
        ylabel=r"$\theta$ [deg]",
        filename="esr_rom_analysis.png",
        prosthesis_subtype=prosthesis_subtype,
        esr_model_type=esr_model_type,
        hlines=[(0, "--", None), (lecomte_max, ":", "Lecomte +14°"), (lecomte_min, ":", "Lecomte -6°")],
        fill=(lecomte_min, lecomte_max))

# Plot knee angle
plot_function(
    steps,
    curves=[(knee_angle_left_deg, "Left / intact"), (knee_angle_right_deg, "Right / prosthesis")],
    title="Knee Joint Angle",
    ylabel="Knee angle [deg]",
    filename="knee_angle.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type)

# Plot knee moment
plot_function(
    steps,
    curves=[(knee_moment_left, "Left / intact"), (knee_moment_right, "Right / prosthesis")],
    title="Knee Joint Moment",
    ylabel="Knee moment [Nm]",
    filename="knee_moment.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type)

# Plot knee power
plot_function(
    steps,
    curves=[(knee_power_left, "Left / intact"), (knee_power_right, "Right / prosthesis")],
    title="Knee Joint Power",
    ylabel="Knee power [W]",
    filename="knee_power.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type,
    hlines=[(0, "--", None)])

# Plot hip angle
plot_function(
    steps,
    curves=[(hip_angle_left_deg, "Left / intact"), (hip_angle_right_deg, "Right / prosthesis")],
    title="Hip Joint Angle",
    ylabel="Hip angle [deg]",
    filename="hip_angle.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type)

# Plot hip moment
plot_function(
    steps,
    curves=[(hip_moment_left, "Left / intact"), (hip_moment_right, "Right / prosthesis")],
    title="Hip Joint Moment",
    ylabel="Hip moment [Nm]",
    filename="hip_moment.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type)

# Plot hip power
plot_function(
    steps,
    curves=[(hip_power_left, "Left / intact"), (hip_power_right, "Right / prosthesis")],
    title="Hip Joint Power",
    ylabel="Hip power [W]",
    filename="hip_power.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type,
    hlines=[(0, "--", None)])


plot_function(
    steps,
    curves=[(grf_left, "Left / intact"), (grf_right, "Right / prosthesis")],
    title="Vertical Ground Reaction Force (toes + calcn)",
    ylabel="Vertical GRF [N]",
    filename="grf_vertical.png",
    prosthesis_subtype=prosthesis_subtype,
    esr_model_type=esr_model_type,
    hlines=[(0, "--", None)])


import pandas as pd

def export_raw_data_to_csv(
        steps,
        esr_theta_deg,
        hip_angle_left_deg, hip_angle_right_deg,
        knee_angle_left_deg, knee_angle_right_deg,
        hip_moment_left, hip_moment_right,
        knee_moment_left, knee_moment_right,
        hip_power_left, hip_power_right,
        knee_power_left, knee_power_right,
        grf_left, grf_right,
        prosthesis_subtype,
        esr_model_type,
        filename="raw_joint_data.csv"):
    """
    Save all raw data to an CSV file.
    Each column corresponds to a raw signal.
    """

    df = pd.DataFrame({
        "step": steps,
        "esr_theta_deg": esr_theta_deg,
        "hip_angle_left_deg": hip_angle_left_deg,
        "hip_angle_right_deg": hip_angle_right_deg,
        "knee_angle_left_deg": knee_angle_left_deg,
        "knee_angle_right_deg": knee_angle_right_deg,
        "hip_moment_left": hip_moment_left,
        "hip_moment_right": hip_moment_right,
        "knee_moment_left": knee_moment_left,
        "knee_moment_right": knee_moment_right,
        "hip_power_left": hip_power_left,
        "hip_power_right": hip_power_right,
        "knee_power_left": knee_power_left,
        "knee_power_right": knee_power_right,
        "grf_left": grf_left,
        "grf_right": grf_right,
    })

    plot_dir = get_plot_dir(prosthesis_subtype, esr_model_type)
    full_path = os.path.join(plot_dir, filename)

    df.to_csv(full_path, index=False)
    print(f"Saved raw data to {full_path}")


if is_esr:
    esr_theta_deg = np.rad2deg(esr_theta)
else:
    esr_theta_deg = np.full_like(steps, np.nan)

export_raw_data_to_csv(
    steps,
    esr_theta_deg,
    hip_angle_left_deg, hip_angle_right_deg,
    knee_angle_left_deg, knee_angle_right_deg,
    hip_moment_left, hip_moment_right,
    knee_moment_left, knee_moment_right,
    hip_power_left, hip_power_right,
    knee_power_left, knee_power_right,
    grf_left, grf_right,
    prosthesis_subtype,
    esr_model_type,
    filename="raw_joint_data.csv"
)