"""
ESR Spring Validation — Linear vs Nonlinear
=============================================
Validates ESR hinge behavior for both linear and nonlinear spring models.
Tests included:
- Test 1: Static torque characteristic
- Test 2: Free dynamic simulation (θ)
- Test 3: Free dynamic simulation (τ & constraint)
- Test 4: Limit behavior
- Test 5: Pseudo-gait with ground contact
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from loco_mujoco import ImitationFactory

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# ─────────────────────────────────────────────────────────────────────────────
# Shared prosthesis configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_PARAMS = {
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

CONFIGS = {
    "linear": {**BASE_PARAMS, "ESR_model_type": "linear"},
    "nonlinear": {**BASE_PARAMS, "ESR_model_type": "nonlinear"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Torque models
# ─────────────────────────────────────────────────────────────────────────────
def tau_linear(theta, L, k_heel, k_keel):
    z = L * np.sin(theta)
    k = k_heel if theta < 0 else k_keel
    Fz = k * z
    return -Fz * L * np.cos(theta)

def tau_nonlinear(theta, L, params):
    z = L * np.sin(theta)
    p = params["heel"] if theta < 0 else params["keel"]
    a, b = p["a"], p["b"]
    Fz = a * np.abs(z) * z + b * z
    return -Fz * L * np.cos(theta)

def compute_tau(theta, model_type, env):
    if model_type == "linear":
        return tau_linear(theta, env.ESR_lever_arm,
                          env.linear_params["k_heel"],
                          env.linear_params["k_keel"])
    else:
        return tau_nonlinear(theta, env.ESR_lever_arm, env.nonlinear_params)

# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Free dynamic simulation (θ)
# ─────────────────────────────────────────────────────────────────────────────
def test_free_sim(model, qpos_adr, dof_adr, compute_tau_fn, env):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[qpos_adr] = np.deg2rad(5.0)
    d.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, d)

    theta_log = []
    tau_log = []
    constraint_log = []

    for _ in range(1000):
        tau = compute_tau_fn(d.qpos[qpos_adr], env.ESR_model_type, env)
        d.qfrc_applied[dof_adr] = tau
        mujoco.mj_step(model, d)

        theta_log.append(np.rad2deg(d.qpos[qpos_adr]))
        tau_log.append(tau)
        constraint_log.append(d.qfrc_constraint[dof_adr])

    return {
        "theta": np.array(theta_log),
        "tau": np.array(tau_log),
        "constraint": np.array(constraint_log),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Limit behavior
# ─────────────────────────────────────────────────────────────────────────────
def test_limit_behavior(model, qpos_adr, dof_adr, compute_tau_fn, env):
    limits = [-6.0, 14.0]
    results = {}

    for lim in limits:
        d = mujoco.MjData(model)
        mujoco.mj_resetData(model, d)
        d.qpos[qpos_adr] = np.deg2rad(lim)
        d.qvel[dof_adr] = 0.0
        mujoco.mj_forward(model, d)

        theta_log = []
        for _ in range(400):
            tau = compute_tau_fn(d.qpos[qpos_adr], env.ESR_model_type, env)
            d.qfrc_applied[dof_adr] = tau
            mujoco.mj_step(model, d)
            theta_log.append(np.rad2deg(d.qpos[qpos_adr]))

        results[lim] = np.array(theta_log)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Pseudo-gait with ground contact
# ─────────────────────────────────────────────────────────────────────────────
def test_pseudo_gait_with_ground(model, qpos_adr, dof_adr, compute_tau_fn, env):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)

    steps = 1200
    tibia_motion = 0.05 * np.sin(np.linspace(0, 4*np.pi, steps))

    theta_log = []
    tau_log = []
    contact_log = []

    for i in range(steps):
        d.qpos[qpos_adr] += tibia_motion[i]
        mujoco.mj_forward(model, d)

        tau = compute_tau_fn(d.qpos[qpos_adr], env.ESR_model_type, env)
        d.qfrc_applied[dof_adr] = tau
        mujoco.mj_step(model, d)

        theta_log.append(np.rad2deg(d.qpos[qpos_adr]))
        tau_log.append(tau)
        contact_log.append(len(d.contact))

    return {
        "theta": np.array(theta_log),
        "tau": np.array(tau_log),
        "contact": np.array(contact_log),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────
def run_tests(model_type, env_params):
    env = ImitationFactory.make(
        "MjxSkeletonMuscleProsthesis",
        **env_params,
        default_dataset_conf=dict(task="walk")
    )

    model = env._model
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
    qpos_adr = model.jnt_qposadr[hinge_id]
    dof_adr = model.jnt_dofadr[hinge_id]

    # Test 1 — static characteristic
    angles_deg = np.linspace(-10, 20, 121)
    tau_vals = np.array([compute_tau(np.deg2rad(a), model_type, env) for a in angles_deg])

    # Test 2 — free dynamic simulation
    free_sim = test_free_sim(model, qpos_adr, dof_adr, compute_tau, env)

    # Test 4 — limit behavior
    limit_behavior = test_limit_behavior(model, qpos_adr, dof_adr, compute_tau, env)

    # Test 5 — pseudo-gait with ground
    gait_ground = test_pseudo_gait_with_ground(model, qpos_adr, dof_adr, compute_tau, env)

    return {
        "angles_deg": angles_deg,
        "tau_vals": tau_vals,
        "free_sim": free_sim,
        "limit_behavior": limit_behavior,
        "gait_ground": gait_ground,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Run both models
# ─────────────────────────────────────────────────────────────────────────────
results = {name: run_tests(name, params) for name, params in CONFIGS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT A — Federphysik (Tests 1, 2, 3)
# ─────────────────────────────────────────────────────────────────────────────
figA, axesA = plt.subplots(3, 2, figsize=(16, 18))
figA.suptitle("ESR Validation — Spring Physics (Linear vs Nonlinear)", fontsize=16, fontweight='bold')

colors = {"linear": ("red", "blue"), "nonlinear": ("darkred", "darkblue")}
labels = {"linear": "Linear", "nonlinear": "Nonlinear"}

for col, (name, res) in enumerate(results.items()):
    heel_color, keel_color = colors[name]

    # Test 1 — static characteristic
    ang = res["angles_deg"]
    tau = res["tau_vals"]
    mask_heel = ang <= 0
    mask_keel = ang >= 0

    ax = axesA[0, col]
    ax.plot(ang[mask_heel], tau[mask_heel], heel_color, linewidth=2.5, label='Heel')
    ax.plot(ang[mask_keel], tau[mask_keel], keel_color, linewidth=2.5, label='Keel')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Test 1 — Static Characteristic ({labels[name]})")
    ax.set_xlabel("θ [°]")
    ax.set_ylabel("τ [Nm]")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    # Test 2 — free dynamic simulation (θ)
    ax = axesA[1, col]
    ax.plot(res["free_sim"]["theta"], keel_color, linewidth=1.0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(14, color='red', linestyle=':', linewidth=2, label='Keel max (+14°)')
    ax.axhline(-6, color='red', linestyle=':', linewidth=2, label='Heel min (-6°)')
    ax.set_title(f"Test 2 — Free Simulation (θ) ({labels[name]})")
    ax.set_xlabel("Step")
    ax.set_ylabel("θ [°]")
    ax.grid(True, linestyle=':', alpha=0.5)

    # Test 3 — applied torque & constraint
    ax = axesA[2, col]
    ax.plot(res["free_sim"]["tau"], color='purple', linewidth=1.0, label='τ applied')
    ax.plot(res["free_sim"]["constraint"], color='red', linewidth=1.0, alpha=0.7, label='qfrc_constraint')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Test 3 — Applied Torque & Constraint ({labels[name]})")
    ax.set_xlabel("Step")
    ax.set_ylabel("[Nm]")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig("esr_validation_spring_physics_2.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PLOT B — Limits & Ground Interaction (Tests 4, 5)
# ─────────────────────────────────────────────────────────────────────────────
figB, axesB = plt.subplots(2, 2, figsize=(16, 14))
figB.suptitle("ESR Validation — Limits & Ground Interaction (Linear vs Nonlinear)", fontsize=16, fontweight='bold')

for col, (name, res) in enumerate(results.items()):
    heel_color, keel_color = colors[name]

    # Test 4 — limit behavior
    ax = axesB[0, col]
    for lim, th in res["limit_behavior"].items():
        ax.plot(th, linewidth=1.5, label=f"{lim}°")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Test 4 — Limit Behavior ({labels[name]})")
    ax.set_xlabel("Step")
    ax.set_ylabel("θ [°]")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    # Test 5 — pseudo-gait with ground
    ax = axesB[1, col]
    ax.plot(res["gait_ground"]["theta"], keel_color, linewidth=1.0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Test 5 — Pseudo-Gait with Ground ({labels[name]})")
    ax.set_xlabel("Step")
    ax.set_ylabel("θ [°]")
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("esr_validation_limits_ground_2.png", dpi=150)
plt.show()
