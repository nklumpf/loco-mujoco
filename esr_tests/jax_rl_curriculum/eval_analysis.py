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

Joint powers are calculated afterwards from the corresponding joint moment
and angular velocity:
    P = M * q_dot

The observed ESR range of motion (ROM) is calculated and compared with
the reference ROM reported by Lecomte et al.

The script uses the MJX evaluation path and must therefore be run without
the --use_mujoco option.

The --record option can optionally be used to record the MJX rendering.

Example:
    python esr_eval.py --path path/to/PPOJax_saved.pkl --n_steps 1000 --record
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
parser = argparse.ArgumentParser(description="Evaluate PPOJax agent and analyse ESR-Prosthesis.")
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--n_steps', type=int, default=1000, help='Number of evaluation steps')
parser.add_argument('--record', action='store_true', help='Record MJX rendering')
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


# Find ESR hinge joint
HINGE_NAME = "esr_hinge_r"  # Name of the ESR hinge joint in the MuJoCo model
# Obtain the numerical MuJoCo joint ID from its name.
hinge_qpos_adr = get_joint_qpos_address(mj_model, HINGE_NAME)
hinge_qvel_adr = get_joint_qvel_address(mj_model, HINGE_NAME)
# print(f"ESR hinge qpos address: {hinge_qpos_adr}")
# print(f"ESR hinge qvel address: {hinge_qvel_adr}")

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
    print(f"{name} torque sensor: "
          f"ID={sensor_id}, "
          f"address={torque_sensor_adr[name]}")

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

    # ESR angle
    esr_theta = float(qpos[hinge_qpos_adr])
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

    # Read torque sensors (here y-component (+1) is choden as sagittal-plane moment)
    sensordata = env_state.data.sensordata[0]
    hip_moment_left   = float(sensordata[torque_sensor_adr["hip_left"] + 1]) 
    hip_moment_right  = float(sensordata[torque_sensor_adr["hip_right"] + 1])
    knee_moment_left  = float(sensordata[torque_sensor_adr["knee_left"] + 1])
    knee_moment_right = float(sensordata[torque_sensor_adr["knee_right"] + 1])

    # Store data
    steps.append(i)
    esr_theta_log.append(esr_theta)
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

    # MJX rendering
    if args.record:
        env.mjx_render(env_state, record=True)

    # Progress output
    if i % 100 == 0:
        print(f"Step {i:5d}: "
              f"ESR theta = "
              f"{np.rad2deg(esr_theta):8.3f} deg")

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
esr_theta_min = esr_theta_deg.min()
esr_theta_max = esr_theta_deg.max()
esr_rom = (esr_theta_max - esr_theta_min)

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


# ============================================================
# Plot
import matplotlib.pyplot as plt
def plot_function(steps, curves, title, ylabel, filename, hlines=None, fill=None):
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
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.show()

# Plot ESR hinge angle
plot_function(
    steps,
    curves=[(esr_theta_deg, "ESR hinge angle")],
    title=f"ESR Hinge Angle — Observed ROM = {esr_rom:.2f}°",
    ylabel=r"$\theta$ [deg]",
    filename="esr_rom_analysis.png",
    hlines=[(0, "--", None), (lecomte_max, ":", "Lecomte +14°"), (lecomte_min, ":", "Lecomte -6°")],
    fill=(lecomte_min, lecomte_max))

# Plot knee angle
plot_function(
    steps,
    curves=[(knee_angle_left_deg, "Left / intact"), (knee_angle_right_deg, "Right / prosthesis")],
    title="Knee Joint Angle",
    ylabel="Knee angle [deg]",
    filename="knee_angle.png")

# Plot knee moment
plot_function(
    steps,
    curves=[(knee_moment_left, "Left / intact"), (knee_moment_right, "Right / prosthesis")],
    title="Knee Joint Moment",
    ylabel="Knee moment [Nm]",
    filename="knee_moment.png")

# Plot knee power
plot_function(
    steps,
    curves=[(knee_power_left, "Left / intact"), (knee_power_right, "Right / prosthesis")],
    title="Knee Joint Power",
    ylabel="Knee power [W]",
    filename="knee_power.png",
    hlines=[(0, "--", None)])

# Plot hip angle
plot_function(
    steps,
    curves=[(hip_angle_left_deg, "Left / intact"), (hip_angle_right_deg, "Right / prosthesis")],
    title="Hip Joint Angle",
    ylabel="Hip angle [deg]",
    filename="hip_angle.png")

# Plot hip moment
plot_function(
    steps,
    curves=[(hip_moment_left, "Left / intact"), (hip_moment_right, "Right / prosthesis")],
    title="Hip Joint Moment",
    ylabel="Hip moment [Nm]",
    filename="hip_moment.png")

# Plot hip power
plot_function(
    steps,
    curves=[(hip_power_left, "Left / intact"), (hip_power_right, "Right / prosthesis")],
    title="Hip Joint Power",
    ylabel="Hip power [W]",
    filename="hip_power.png",
    hlines=[(0, "--", None)])