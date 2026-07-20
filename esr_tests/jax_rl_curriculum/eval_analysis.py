"""
ESR Gait Analysis — Trained RL Agent
======================================
Loads a trained PPOJax checkpoint, runs evaluation, logs ESR hinge
angle (theta) and torque, then plots ROM analysis vs Lecomte reference.

Usage:
    conda activate loconaomi
    python esr_eval_analysis.py --path PATH/TO/PPOJax_saved.pkl
    python esr_eval_analysis.py --path PATH/TO/PPOJax_saved.pkl --use_mujoco
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
os.environ["MUJOCO_GL"] = "egl"  # headless rendering

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='ESR Gait Analysis — Trained RL Agent')
parser.add_argument('--path',       type=str, required=True,
                    help='Path to PPOJax_saved.pkl checkpoint')
parser.add_argument('--use_mujoco', action='store_true',
                    help='Use MuJoCo CPU path instead of MJX')
parser.add_argument('--n_steps',    type=int, default=1000,
                    help='Number of steps to evaluate')
parser.add_argument('--n_envs',     type=int, default=1,
                    help='Number of parallel envs (MJX only)')
args = parser.parse_args()

# ── Load agent ────────────────────────────────────────────────────────────────
print(f"Loading agent from: {args.path}")
agent_conf, agent_state = PPOJax.load_agent(args.path)
config = agent_conf.config

# ── Build environment ─────────────────────────────────────────────────────────
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

domain_randomization_params = {
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {},
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {
        "pylon_socket": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]},
        "talus":        {"x": [-0.0, 0.0], "z": [-0.0, 0.0]},
    },
    "randomize_prosthesis_body_orientation": True,
    "prosthesis_body_orientation_range": {
        "pylon_socket": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
        "talus":        {"x": [-0.0, 0.0], "z": [-0.0, 0.0]},
    },
}

OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True   # no viewer
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"

env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    domain_randomization_type=config.randomization_config["randomization_type"],
    domain_randomization_params=domain_randomization_params,
)
print("Environment created.")

# ── ESR hinge IDs ─────────────────────────────────────────────────────────────
model          = env._model
hinge_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
hinge_qpos_adr = model.jnt_qposadr[hinge_id]
hinge_dof_adr  = model.jnt_dofadr[hinge_id]
L              = env.ESR_lever_arm
k_heel         = env.linear_params["k_heel"]
k_keel         = env.linear_params["k_keel"]

print(f"ESR hinge id={hinge_id}  qpos={hinge_qpos_adr}  dof={hinge_dof_adr}")
print(f"k_heel={k_heel:.1f} N/m  k_keel={k_keel:.1f} N/m  L={L*1000:.1f} mm")

# ── Logging ───────────────────────────────────────────────────────────────────
log_theta = []
log_tau   = []
step_count = [0]

def log_step():
    """Read theta and tau from env._data (MuJoCo CPU path)."""
    if hasattr(env, '_data') and env._data is not None:
        theta = float(env._data.qpos[hinge_qpos_adr])
        z     = L * np.sin(theta)
        k     = k_heel if theta < 0 else k_keel
        tau   = -k * z * L * np.cos(theta)
        log_theta.append(np.rad2deg(theta))
        log_tau.append(tau)

_original_step = env.step

def _step_with_logging(action):
    result = _original_step(action)
    step_count[0] += 1
    log_step()
    if step_count[0] % 100 == 0:
        theta_str = f"{log_theta[-1]:.2f}°" if log_theta else "N/A"
        print(f"  Step {step_count[0]:4d}/{args.n_steps}  θ={theta_str}")
    return result

env.step = _step_with_logging

# ── Run evaluation ────────────────────────────────────────────────────────────
print(f"\nRunning evaluation ({args.n_steps} steps, "
      f"{'MuJoCo CPU' if args.use_mujoco else 'MJX GPU'})...")

if args.use_mujoco:
    PPOJax.play_policy_mujoco(
        env, agent_conf, agent_state,
        deterministic=False,
        n_steps=args.n_steps,
        record=False,          # no video = no viewer blocking
        train_state_seed=0
    )
else:
    PPOJax.play_policy(
        env, agent_conf, agent_state,
        deterministic=False,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        record=False,          # no video = no viewer blocking
        train_state_seed=0
    )

env.step = _original_step

# ── Check if logging worked ───────────────────────────────────────────────────
print(f"\nLogged {len(log_theta)} steps.")

if len(log_theta) == 0:
    print("\nWARNING: No data logged!")
    print("env.step was not called by play_policy.")
    print("This happens with the MJX path.")
    print("Try: python esr_eval_analysis.py --path PATH --use_mujoco")
    exit(0)

# ── Statistics ────────────────────────────────────────────────────────────────
log_theta = np.array(log_theta)
log_tau   = np.array(log_tau)

print(f"\n{'='*60}")
print("ROM STATISTICS")
print(f"{'='*60}")
print(f"  Min θ  : {log_theta.min():.1f}°  (Lecomte ref: -6°)")
print(f"  Max θ  : {log_theta.max():.1f}°  (Lecomte ref: +14°)")
print(f"  ROM    : {log_theta.max()-log_theta.min():.1f}°  (Lecomte ref: 20°)")
print(f"  Mean θ : {log_theta.mean():.1f}°")
print(f"\n  Heel phase (θ<0): {(log_theta < 0).sum()} steps  "
      f"({(log_theta < 0).mean()*100:.1f}%)")
print(f"  Keel phase (θ>0): {(log_theta > 0).sum()} steps  "
      f"({(log_theta > 0).mean()*100:.1f}%)")
print(f"\n  Applied torque range: [{log_tau.min():.1f}, {log_tau.max():.1f}] Nm")
print(f"  Mean |τ|: {np.abs(log_tau).mean():.1f} Nm")

# ── Plots ─────────────────────────────────────────────────────────────────────
steps = np.arange(len(log_theta))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("ESR Gait Analysis — Trained RL Agent", fontsize=13, fontweight='bold')

# ROM over time
axes[0].plot(steps, log_theta, 'b-', linewidth=0.8, label='ESR θ')
axes[0].axhline(y=0,   color='gray',   linestyle='--', alpha=0.5, label='Neutral')
axes[0].axhline(y=14,  color='orange', linestyle=':', linewidth=2,
                label='Lecomte keel max (+14°)')
axes[0].axhline(y=-6,  color='red',    linestyle=':', linewidth=2,
                label='Lecomte heel min (-6°)')
axes[0].fill_between(steps, -6, 14, alpha=0.05, color='green',
                     label='Lecomte ROM range')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('θ [°]')
axes[0].set_title(f'ESR Hinge Angle  '
                  f'(min={log_theta.min():.1f}°, max={log_theta.max():.1f}°, '
                  f'ROM={log_theta.max()-log_theta.min():.1f}°  |  '
                  f'Lecomte ref: ROM=20°)')
axes[0].legend(fontsize=8)
axes[0].grid(True, linestyle=':', alpha=0.5)

# Torque over time
axes[1].plot(steps, log_tau, color='purple', linewidth=0.8, label='τ ESR spring')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('τ [Nm]')
axes[1].set_title(f'Applied Torque  '
                  f'(range: [{log_tau.min():.1f}, {log_tau.max():.1f}] Nm)')
axes[1].legend(fontsize=8)
axes[1].grid(True, linestyle=':', alpha=0.5)

# τ-θ scatter (spring characteristic in use)
sc = axes[2].scatter(log_theta, log_tau, c=steps, cmap='viridis', s=1, alpha=0.5)
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(x=14,  color='orange', linestyle=':', alpha=0.7,
                label='Lecomte keel max')
axes[2].axvline(x=-6,  color='red',    linestyle=':', alpha=0.7,
                label='Lecomte heel min')
plt.colorbar(sc, ax=axes[2], label='Step')
axes[2].set_xlabel('θ [°]')
axes[2].set_ylabel('τ [Nm]')
axes[2].set_title('τ–θ Curve (spring characteristic in use)')
axes[2].legend(fontsize=8)
axes[2].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("esr_rom_analysis.png", dpi=150)
print("\nSaved: esr_rom_analysis.png")
plt.show()