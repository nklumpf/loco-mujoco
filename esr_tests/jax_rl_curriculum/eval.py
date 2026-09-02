import os

# Uncomment the following lines to force JAX to use CPU instead of GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
# Uncomment the following line to force JAX to use CPU instead of GPU
# jax.config.update('jax_platform_name', 'cpu')

import argparse
import numpy as np
import matplotlib.pyplot as plt

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# headless
os.environ["MUJOCO_GL"] = "egl"


# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
domain_randomization_type = config.randomization_config["randomization_type"]
domain_randomization_params =  {
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {}, 
    "randomize_prosthesis_joint_stiffness": False, 
    "prosthesis_joint_stiffness_range": {}, 
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {"pylon_socket": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]}, "talus": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]}},
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
    },
}

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params,
                   domain_randomization_type=domain_randomization_type,
                    domain_randomization_params=domain_randomization_params)

env._esr_log_enabled = True
env._esr_log_theta = []
env._esr_log_tau = []

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=1000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                       train_state_seed=0)


theta = np.asarray(env._esr_log_theta)
tau = np.asarray(env._esr_log_tau)

print("\nESR LOGGING")
print(f"Number of samples: {len(theta)}")
print(f"Theta min: {np.rad2deg(theta).min():.3f} deg")
print(f"Theta max: {np.rad2deg(theta).max():.3f} deg")
print(f"ROM: {np.rad2deg(theta).max() - np.rad2deg(theta).min():.3f} deg")

print(f"Tau min: {tau.min():.3f} Nm")
print(f"Tau max: {tau.max():.3f} Nm")

steps = np.arange(len(theta))

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

theta_deg = np.rad2deg(theta)

# ---------------------------------------------------------
# Theta over time
# ---------------------------------------------------------

axes[0].plot(steps, theta_deg)

axes[0].axhline(0, linestyle="--", alpha=0.5)
axes[0].axhline(14, linestyle=":", alpha=0.7)
axes[0].axhline(-6, linestyle=":", alpha=0.7)

axes[0].set_ylabel(r"$\theta$ [deg]")
axes[0].set_xlabel("Evaluation step")
axes[0].set_title(
    f"ESR Hinge Angle — ROM = "
    f"{theta_deg.max() - theta_deg.min():.2f}°"
)

axes[0].grid(True)


# ---------------------------------------------------------
# Torque over time
# ---------------------------------------------------------

axes[1].plot(steps, tau)

axes[1].axhline(0, linestyle="--", alpha=0.5)

axes[1].set_ylabel(r"$\tau_{\mathrm{ESR}}$ [Nm]")
axes[1].set_xlabel("Evaluation step")
axes[1].set_title("ESR Torque")

axes[1].grid(True)


# ---------------------------------------------------------
# Torque-angle curve
# ---------------------------------------------------------

axes[2].scatter(theta_deg, tau, s=2, alpha=0.5)

axes[2].set_xlabel(r"$\theta$ [deg]")
axes[2].set_ylabel(r"$\tau_{\mathrm{ESR}}$ [Nm]")
axes[2].set_title("ESR Torque–Angle Relationship")

axes[2].grid(True)

plt.tight_layout()

plt.savefig(
    f"esr_analysis_{env.ESR_model_type}.png",
    dpi=150
)

plt.show()