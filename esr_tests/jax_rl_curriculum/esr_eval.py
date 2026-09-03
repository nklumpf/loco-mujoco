"""
This script evaluates a trained PPOJax locomotion policy using the MJX
simulation and analyses the motion of the ESR prosthesis.

During the evaluation, the ESR hinge angle is read directly from the MJX
simulation state (qpos). The observed range of motion (ROM) is then
calculated and compared with the reference ROM reported by Lecomte et al.

The script uses the MJX evaluation path and must therefore be run without
the --use_mujoco option.

The --record option can optionally be used to record the MJX rendering.

Example:
  python esr_eval.py --path path/to/PPOJax_saved.pkl --n_steps 1000 --record

"""

import os

# Uncomment the following lines to force JAX to use CPU instead of GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
# Uncomment the following line to force JAX to use CPU instead of GPU
# jax.config.update('jax_platform_name', 'cpu')

import argparse
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True")

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
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2" # nicer looking than GoalTrajMimic

print("\nCreating environment...")
env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    domain_randomization_type=domain_randomization_type,
    domain_randomization_params=domain_randomization_params)

# ESR logging variables
env._esr_log_enabled = True
env._esr_log_theta = []
env._esr_log_tau = []


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
        for name in ["mj_model","model","_model"]:
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

# Find ESR hinge joint
HINGE_NAME = "esr_hinge_r"  # Name of the ESR hinge joint in the MuJoCo model 
# Obtain the numerical MuJoCo joint ID from its name.
hinge_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, HINGE_NAME)
if hinge_id < 0:
    raise RuntimeError(
        f"Could not find joint '{HINGE_NAME}' "
        "in the MuJoCo model.")
# Get the corresponding position index in qpos.
hinge_qpos_adr = int(mj_model.jnt_qposadr[hinge_id])
print(f"ESR hinge joint ID: {hinge_id}")
print(f"ESR hinge qpos address: {hinge_qpos_adr}")

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
    The running observation statistics are updated in same way as in
    PPOJax.play_policy().
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


# ============================================================
# Evaluation
print("\nStarting evaluation...")

theta_log = []
steps = []

for i in range(args.n_steps):

    # Random key
    rng, _rng = jax.random.split(rng)

    # Policy action
    action, train_state = plcy_call(train_state, obs, _rng)
    action = jnp.atleast_2d(action)

    # MJX step
    (obs, reward, absorbing, done, info, env_state) = env.step(env_state, action)

    # Read ESR hinge angle directly from MJX qpos
    theta = env_state.data.qpos[0, hinge_qpos_adr]
    theta = float(theta)

    # Store data
    theta_log.append(theta)
    steps.append(i)

    # MJX rendering
    if args.record:
        env.mjx_render(env_state, record=True)

    # Progress output
    if i % 100 == 0:
        print(f"Step {i:5d}: "
              f"theta = "
              f"{np.rad2deg(theta):8.3f} deg")

env.stop()

# Convert recorded data to NumPy arrays
theta = np.asarray(theta_log)
steps = np.asarray(steps)
theta_deg = np.rad2deg(theta)

# ROM
theta_min = theta_deg.min()
theta_max = theta_deg.max()
rom = theta_max - theta_min

print("\n")
print("=" * 60)
print("ESR ROM ANALYSIS")
print("=" * 60)

print(f"Theta min: {theta_min:.3f} deg")
print(f"Theta max: {theta_max:.3f} deg")
print(f"ROM:       {rom:.3f} deg")
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
plt.figure(figsize=(12, 6))
plt.plot(steps, theta_deg, linewidth=0.8, label="ESR hinge angle")
plt.axhline(0, linestyle="--", alpha=0.5)
plt.axhline(lecomte_max, linestyle=":", linewidth=2, label="Lecomte +14°")
plt.axhline(lecomte_min, linestyle=":", linewidth=2, label="Lecomte -6°")

plt.fill_between(steps, lecomte_min, lecomte_max, alpha=0.05)
plt.xlabel("Evaluation step")
plt.ylabel(r"$\theta$ [deg]")
plt.title(f"ESR Hinge Angle — " f"Observed ROM = {rom:.2f}°")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("esr_rom_analysis.png", dpi=150)
plt.show()