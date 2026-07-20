import os
import jax
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from loco_mujoco import ImitationFactory

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

env_params = {
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial",
    "prosthesis_subtype": "ESR",
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2,
    "reattach_muscles": {"med_gas": [0, 0, 0]},
    "remove_joint_names": ["subtalar_angle"],
    "multi_contact_geom_type": "2boxes",
    "contact_geom_solref": [-900, -300],
    "socket_joint_dofs": [
        'socket_tx', 'socket_ty', 'socket_tz',
        'socket_flexion', 'socket_adduction', 'socket_rotation'
    ],
    "visualize_prosthesis": True,
}

env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    **env_params,
    default_dataset_conf=dict(task="walk")
)

# ── Hinge qpos address ────────────────────────────────────────────────────────
if hasattr(env, 'esr_hinge_info') and "_r" in env.esr_hinge_info:
    hinge_qpos_adr = env.esr_hinge_info["_r"]["hinge_qpos"]
else:
    hinge_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
    if hinge_id == -1:
        raise RuntimeError("esr_hinge_r not found. Set visualize_prosthesis=True.")
    hinge_qpos_adr = env._model.jnt_qposadr[hinge_id]
print(f"Hinge qpos address: {hinge_qpos_adr}")

# ── MJX Setup ─────────────────────────────────────────────────────────────────
key    = jax.random.key(0)
n_envs = 10
keys   = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

rng_reset  = jax.jit(jax.vmap(env.mjx_reset))
rng_step   = jax.jit(jax.vmap(env.mjx_step))
rng_sample = jax.jit(jax.vmap(env.sample_action_space))

state = rng_reset(env_keys)

# Verify qpos access
q_test = np.array(state.data.qpos)
print(f"qpos shape: {q_test.shape}  (n_envs={n_envs}, nq={q_test.shape[1]})")
print(f"theta at reset: {np.rad2deg(q_test[:, hinge_qpos_adr])}")

# ── Theta logging ─────────────────────────────────────────────────────────────
n_steps   = 500
log_theta = []

print(f"\nRunning {n_steps} steps × {n_envs} envs...")
for i in range(n_steps):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample(action_keys)
    state  = rng_step(state, action)
    q = np.array(state.data.qpos)
    log_theta.append(np.rad2deg(q[:, hinge_qpos_adr]))

log_theta = np.array(log_theta)  # (n_steps, n_envs)
all_theta = log_theta.flatten()

# ── Statistics ────────────────────────────────────────────────────────────────
print(f"\n=== THETA STATISTICS ===")
print(f"ROM:  {all_theta.max()-all_theta.min():.1f}°")
print(f"Min:  {all_theta.min():.1f}°")
print(f"Max:  {all_theta.max():.1f}°")
print(f"Mean: {all_theta.mean():.1f}°")
print(f"\nLecomte Vari-Flex XC reference: ROM=20°, heel=-6°, keel=+14°")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for env_idx in range(min(n_envs, 5)):
    axes[0].plot(log_theta[:, env_idx], linewidth=0.8, alpha=0.7,
                 label=f'Env {env_idx+1}')
axes[0].axhline(y=0,    color='gray',   linestyle='--', alpha=0.5, label='Neutral')
axes[0].axhline(y=14.0, color='orange', linestyle=':', linewidth=2,
                label='Lecomte keel max (+14°)')
axes[0].axhline(y=-6.0, color='red',    linestyle=':', linewidth=2,
                label='Lecomte heel min (-6°)')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('θ [°]')
axes[0].set_title('ESR Hinge Angle — MJX Simulation (random actions)')
axes[0].legend(fontsize=8)
axes[0].grid(True, linestyle=':', alpha=0.5)

axes[1].hist(all_theta, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(x=0,    color='gray',   linestyle='--', alpha=0.7, label='Neutral')
axes[1].axvline(x=14.0, color='orange', linestyle=':', linewidth=2,
                label='Lecomte keel max (+14°)')
axes[1].axvline(x=-6.0, color='red',    linestyle=':', linewidth=2,
                label='Lecomte heel min (-6°)')
axes[1].set_xlabel('θ [°]')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution (all envs × all steps)')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("esr_hinge_theta_mjx.png", dpi=150)
print("Saved: esr_hinge_theta_mjx.png")
plt.show()

hinge_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
print(f"limited: {env._model.jnt_limited[hinge_id]}")
print(f"range:   {np.rad2deg(env._model.jnt_range[hinge_id])}")