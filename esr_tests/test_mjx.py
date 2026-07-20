"""
ESR Hinge — MJX Sign Check + Viewer (combined)
=================================================
1. Checks sign and convergence (zero action, isolated hinge)
2. Opens the Viewer to visually observe the hinge's motion

Important: zero action = no trained agent; the human falls over.
         This is intentional—we are only isolating the spring dynamics.
"""
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from loco_mujoco import ImitationFactory

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# ── Environment ───────────────────────────────────────────────────────────────
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
    "visualize_prosthesis": True,
}

env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    **env_params,
    default_dataset_conf=dict(task="walk")
)

model          = env._model
hinge_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
hinge_qpos_adr = model.jnt_qposadr[hinge_id]
hinge_dof_adr  = model.jnt_dofadr[hinge_id]

# ── Model info ────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL INFO")
print("=" * 60)
print(f"  k_heel    : {env.linear_params['k_heel']:.1f} N/m")
print(f"  k_keel    : {env.linear_params['k_keel']:.1f} N/m")
print(f"  Lever arm : {env.ESR_lever_arm*1000:.1f} mm")
print(f"  limited   : {model.jnt_limited[hinge_id]}")
print(f"  range     : {np.rad2deg(model.jnt_range[hinge_id])} °")
print(f"  damping   : {model.dof_damping[hinge_dof_adr]:.2f} Nm·s/rad")
print(f"  stiffness : {model.jnt_stiffness[hinge_id]:.2f} Nm/rad")

# ── MJX Setup ─────────────────────────────────────────────────────────────────
key    = jax.random.key(0)
n_envs = 1   # single env for viewer sync
keys   = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step  = jax.jit(jax.vmap(env.mjx_step))

state = rng_reset(env_keys)

# Zero action — isolates spring, no muscle forces
act_size = model.nu
action   = jnp.zeros((n_envs, act_size))

# ── Helper ────────────────────────────────────────────────────────────────────
def set_theta(state, theta_deg):
    qpos = np.array(state.data.qpos)
    qpos[:, hinge_qpos_adr] = np.deg2rad(theta_deg)
    new_data = state.data.replace(qpos=jnp.array(qpos))
    return state.replace(data=new_data)

def get_theta(state):
    return float(np.rad2deg(np.array(state.data.qpos)[0, hinge_qpos_adr]))


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Sign Check + Convergence (20 steps each)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1: Sign Check + Convergence (20 steps, zero action)")
print("=" * 60)
print("Note: system is unstable without trained agent (human falls)")
print("      → only ESR hinge dynamics are meaningful here\n")

n_steps    = 20
log_keel   = [10.0]
log_heel   = [-5.0]

# Single step sign check
for theta_deg, label, expect_restoring in [
    (+10.0, "Keel (+10°)", True),
    (-5.0,  "Heel (-5°)",  True),
]:
    s       = set_theta(state, theta_deg)
    s_after = rng_step(s, action)
    t_after = get_theta(s_after)
    ok      = abs(t_after) < abs(theta_deg)
    print(f"  {label}: {theta_deg:+.1f}° → {t_after:+.3f}°  "
          f"{'✓ restoring' if ok else '✗ WRONG SIGN'}")

# Multi-step convergence
print("\n  Keel multi-step (+10°):")
s = set_theta(state, +10.0)
for i in range(n_steps):
    s = rng_step(s, action)
    theta = get_theta(s)
    log_keel.append(theta)
    if i < 5 or i == n_steps - 1:
        print(f"    Step {i+1:2d}: {theta:+.3f}°")
    elif i == 5:
        print(f"    ...")

print(f"\n  Heel multi-step (-5°):")
s = set_theta(state, -5.0)
for i in range(n_steps):
    s = rng_step(s, action)
    theta = get_theta(s)
    log_heel.append(theta)
    if i < 5 or i == n_steps - 1:
        print(f"    Step {i+1:2d}: {theta:+.3f}°")
    elif i == 5:
        print(f"    ...")

keel_ok = abs(log_keel[-1]) < 10.0
heel_ok = abs(log_heel[-1]) < 5.0
print(f"\n  Keel converges: {'✓' if keel_ok else '✗'}")
print(f"  Heel converges: {'✓' if heel_ok else '✗'}")

# Joint limit check
print("\n  Joint limit check (+30°, beyond limit):")
s = set_theta(state, +30.0)
thetas_limit = []
for i in range(5):
    s = rng_step(s, action)
    theta = get_theta(s)
    thetas_limit.append(theta)
    print(f"    Step {i+1}: {theta:+.3f}°")
limits_ok = all(abs(t) < 100.0 for t in thetas_limit)
print(f"  Limits effective (no explosion): {'✓' if limits_ok else '✗ ADD CLIP!'}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
steps = np.arange(0, n_steps + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ESR Hinge — MJX Sign Check (zero action, isolated spring)",
             fontsize=12, fontweight='bold')

for ax, log, start, label, color, ref_y, ref_label in [
    (axes[0], log_keel, +10, "Keel: Start +10°", 'blue',
     14, 'Lecomte keel max (+14°)'),
    (axes[1], log_heel, -5,  "Heel: Start -5°",  'red',
     -6, 'Lecomte heel min (-6°)'),
]:
    ax.plot(steps, log, '-o', linewidth=2, markersize=5,
            color=color, label='θ(t)')
    ax.axhline(y=0,     color='gray',   linestyle='--', alpha=0.6, label='Neutral')
    ax.axhline(y=start, color=color,    linestyle=':',  alpha=0.4,
               label=f'Start ({start:+d}°)')
    ax.axhline(y=ref_y, color='orange' if ref_y > 0 else 'red',
               linestyle=':', linewidth=2, label=ref_label)
    ax.fill_between(steps, -6, 14, alpha=0.07, color='green',
                    label='Lecomte ROM [-6°, +14°]')
    ax.set_xlabel('Step')
    ax.set_ylabel('θ [°]')
    ax.set_title(f'{label}\nFinal: {log[-1]:.2f}°  '
                 f'{"✓ converging" if abs(log[-1]) < abs(start) else "✗ diverging"}')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("esr_mjx_sign_check.png", dpi=150)
print("\nSaved: esr_mjx_sign_check.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Viewer — live observation of hinge motion
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Viewer (close window to exit)")
print("=" * 60)
print("Showing: zero action, start at +10°")
print("Watch the forefoot — it should oscillate and slowly converge to 0°")
print("Note: human will fall (no trained agent) — focus on the ESR forefoot\n")

# Reset and set initial theta
state = rng_reset(env_keys)
state = set_theta(state, +10.0)

# MuJoCo data for viewer
data_viewer = mujoco.MjData(model)
mujoco.mj_resetData(model, data_viewer)

step = 0
log_theta_viewer = []

with mujoco.viewer.launch_passive(model, data_viewer) as viewer:
    # Focus camera on the foot
    viewer.cam.lookat[:] = [0.0, -0.085, 0.08]
    viewer.cam.distance  = 0.8
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -5
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]   = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True

    while viewer.is_running():
        # MJX step
        state = rng_step(state, action)
        step += 1

        # Sync MJX → MuJoCo viewer
        data_viewer.qpos[:] = np.array(state.data.qpos)[0]
        data_viewer.qvel[:] = np.array(state.data.qvel)[0]
        mujoco.mj_forward(model, data_viewer)

        theta = data_viewer.qpos[hinge_qpos_adr]
        log_theta_viewer.append(np.rad2deg(theta))

        if step % 20 == 0:
            print(f"  Step {step:4d}: θ={np.rad2deg(theta):+.2f}°")

        viewer.sync()
        time.sleep(0.02)

# Plot theta from viewer session
if len(log_theta_viewer) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(log_theta_viewer, 'b-', linewidth=1.0, label='θ(t) viewer session')
    plt.axhline(y=0,  color='gray',   linestyle='--', alpha=0.5, label='Neutral')
    plt.axhline(y=14, color='orange', linestyle=':', linewidth=2,
                label='Lecomte keel max (+14°)')
    plt.axhline(y=-6, color='red',    linestyle=':', linewidth=2,
                label='Lecomte heel min (-6°)')
    plt.xlabel('Step')
    plt.ylabel('θ [°]')
    plt.title('ESR Hinge θ — Viewer Session')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("esr_viewer_theta.png", dpi=150)
    print(f"\nSaved: esr_viewer_theta.png")
    print(f"Viewer session ROM: {max(log_theta_viewer)-min(log_theta_viewer):.1f}°")
    plt.show()

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Sign correct (single step):      {'✓' if keel_ok or heel_ok else '✗'}")
print(f"  Keel converges (20 steps):       {'✓' if keel_ok else '✗'}")
print(f"  Heel converges (20 steps):       {'✓' if heel_ok else '✗'}")
print(f"  No explosion beyond limits:      {'✓' if limits_ok else '✗ → add clip'}")
print()
if keel_ok and heel_ok and limits_ok:
    print("  → ALL CHECKS PASSED.")
else:
    print("  → Think about it again.")