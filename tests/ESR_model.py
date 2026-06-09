"""
ESR Force Model Test Suite
==========================
Tests the Rigney (2018) force model implementation in MjxSkeletonMuscleProsthesis.

Tests:
    1. Zero-force test:     Z=0, Y=0 -> Fz≈0, Fy≈0
    2. Monotonicity test:   increasing Z -> increasing Fz
    3. Sign test:           Z<0 -> Fz>0 (restoring), Z>0 -> Fz<0
    4. Y-coupling test:     Y variation -> Fy response (distal displacement only)
    5. Alpha test:          alpha is in plausible range during normal stance
    6. Force-displacement plot: visual check of curve shape

Usage:
    python test_esr_force_model.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from loco_mujoco import ImitationFactory
import mujoco
from mujoco import mjx


# ============================================================
# Setup
# ============================================================

ESR_MODEL_TYPE = "distal_displacement" #"linear_elastic"  # or "distal_displacement"

env_params = {
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial",
    "prosthesis_subtype": "ESR",
    "ESR_model_type": ESR_MODEL_TYPE,
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2,
    "reattach_muscles": {"med_gas": [0, 0, 0]},
    "remove_joint_names": ["subtalar_angle"],
}

print("Creating ESR environment...")
env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    **env_params,
    default_dataset_conf=dict(task="walk"),
)

key = jax.random.key(0)
state = env.mjx_reset(key)
data = state.data
info = env.esr_joint_info["_r"]

SIDE = "_r"
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def set_displacement(data, z_mm=0.0, y_mm=0.0):
    """Set socket_ty (Z) and socket_tx (Y) displacements."""
    qpos = data.qpos
    qpos = qpos.at[info["ty_qpos"]].set(z_mm / 1000.0)
    qpos = qpos.at[info["tx_qpos"]].set(y_mm / 1000.0)
    return data.replace(qpos=qpos)


def get_forces(data):
    """Apply ESR force model and return Fz, Fy."""
    # Reset qfrc_applied to isolate ESR contribution
    data_clean = data.replace(
        qfrc_applied=data.qfrc_applied * 0.0
    )
    data_out = env.compute_ESR_qfrc(data_clean)
    Fz = float(data_out.qfrc_applied[info["ty_dof"]])
    Fy = float(data_out.qfrc_applied[info["tx_dof"]])
    return Fz, Fy


# ============================================================
# Test 1: Zero-force test
# ============================================================

print("\n" + "="*50)
print("TEST 1: Zero-force at Z=0, Y=0")
print("="*50)
print("Expected: Fz ≈ 0, Fy ≈ 0")

data_zero = set_displacement(data, z_mm=0.0, y_mm=0.0)
Fz, Fy = get_forces(data_zero)
alpha = float(env.get_pylon_alpha(data_zero, SIDE))

print(f"  Fz = {Fz:.4f} N")
print(f"  Fy = {Fy:.4f} N")
print(f"  alpha = {np.rad2deg(alpha):.2f} deg")

tol = 1.0  # N
result = PASS if abs(Fz) < tol and abs(Fy) < tol else FAIL
print(f"  Result: {result}")


# ============================================================
# Test 2: Monotonicity test
# ============================================================

print("\n" + "="*50)
print("TEST 2: Monotonicity (increasing compression -> increasing force)")
print("="*50)
print("Expected: |Fz| increases monotonically with |Z|")
print(f"{'Z_mm':>8} {'Fz_N':>10} {'Fy_N':>10} {'alpha_deg':>12}")
print("-" * 45)

z_test_values = [0, -5, -10, -15, -20, -25, -30]
fz_values = []

for z_mm in z_test_values:
    d = set_displacement(data, z_mm=z_mm)
    Fz, Fy = get_forces(d)
    alpha = float(env.get_pylon_alpha(d, SIDE))
    fz_values.append(Fz)
    print(f"{z_mm:8.1f} {Fz:10.2f} {Fy:10.2f} {np.rad2deg(alpha):12.2f}")

# Check monotonicity for compression (z < 0 should give Fz > 0 and increasing)
compression_forces = [fz_values[i] for i in range(1, len(z_test_values))]
is_monotone = all(
    compression_forces[i] >= compression_forces[i-1]
    for i in range(1, len(compression_forces))
)
result = PASS if is_monotone else FAIL
print(f"  Monotone: {result}")


# ============================================================
# Test 3: Sign test
# ============================================================

print("\n" + "="*50)
print("TEST 3: Sign test")
print("="*50)
print("Expected: Z<0 (compression) -> Fz>0 (restoring upward)")
print("          Z>0 (tension)      -> Fz<0")

for z_mm in [-10.0, 10.0]:
    d = set_displacement(data, z_mm=z_mm)
    Fz, Fy = get_forces(d)
    expected_positive = z_mm < 0
    correct_sign = (Fz > 0) == expected_positive
    result = PASS if correct_sign else FAIL
    print(f"  Z={z_mm:+.1f}mm -> Fz={Fz:+.2f}N  {result}")


# ============================================================
# Test 4: Y-coupling test
# ============================================================

print("\n" + "="*50)
print("TEST 4: Y-coupling")
print("="*50)

if ESR_MODEL_TYPE == "linear_elastic":
    print("Linear elastic: Fy should be ≈ 0 regardless of Y")
    print(f"{'Y_mm':>8} {'Fz_N':>10} {'Fy_N':>10}")
    print("-" * 32)
    all_fy_zero = True
    for y_mm in [0, 5, 10, 20]:
        d = set_displacement(data, z_mm=-10.0, y_mm=y_mm)
        Fz, Fy = get_forces(d)
        print(f"{y_mm:8.1f} {Fz:10.2f} {Fy:10.2f}")
        if abs(Fy) > 1.0:
            all_fy_zero = False
    result = PASS if all_fy_zero else FAIL
    print(f"  Fy ≈ 0: {result}")

elif ESR_MODEL_TYPE == "distal_displacement":
    print("Distal displacement: Fy should respond to Y variation")
    print(f"{'Y_mm':>8} {'Fz_N':>10} {'Fy_N':>10}")
    print("-" * 32)
    fy_values = []
    for y_mm in [0, 5, 10, 15, 20]:
        d = set_displacement(data, z_mm=-10.0, y_mm=y_mm)
        Fz, Fy = get_forces(d)
        fy_values.append(Fy)
        print(f"{y_mm:8.1f} {Fz:10.2f} {Fy:10.2f}")
    has_fy_response = any(abs(f) > 0.1 for f in fy_values[1:])
    result = PASS if has_fy_response else FAIL
    print(f"  Fy responds to Y: {result}")


# ============================================================
# Test 5: Alpha plausibility
# ============================================================
z_sweep = np.linspace(-10, 0, 100)
print("\n" + "="*50)
print("TEST 5: Alpha plausibility")
print("="*50)

alpha_rad = float(env.get_pylon_alpha(data, SIDE))
alpha_deg = np.rad2deg(alpha_rad)
print(f"  alpha = {alpha_deg:.2f} deg")
print(f"  (Rigney tested at 12° and 22°)")

# ============================================================
# Test 5b: Alpha variation via knee angle
# ============================================================

print("\n" + "="*50)
print("TEST 5b: Alpha variation via knee angle")
print("="*50)
print("Rigney tested at 12° and 22°")
print(f"{'target_alpha':>12} {'knee_deg':>10} {'alpha_actual':>14} {'Fz@Z=-10mm':>12} {'Fy@Z=-10mm':>12}")
print("-" * 65)

knee_id = mujoco.mj_name2id(
    env._model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
knee_qpos = env._model.jnt_qposadr[knee_id]

# From joint test:
# knee=0°  -> alpha=-7.66°
# knee=10° -> alpha=-17.67°
# So: alpha = -7.66 - knee  =>  knee = -(target_alpha + 7.66)
# Note: negative alpha = pylon tilted forward (anterior) - physically correct during gait

target_alphas = [0, 5, 12, 17, 22]

fz_at_rigney = {}

for target_alpha in target_alphas:
    knee_deg = -(target_alpha + 7.66)

    d = data.replace(
        qpos=data.qpos.at[knee_qpos].set(np.deg2rad(knee_deg))
    )
    d = mjx.forward(env.sys, d)

    # Set Z=-30mm
    d = d.replace(
        qpos=d.qpos.at[info["ty_qpos"]].set(-0.01)
    )

    alpha_actual = np.rad2deg(float(env.get_pylon_alpha(d, SIDE)))
    Fz, Fy = get_forces(d)
    fz_at_rigney[target_alpha] = Fz

    print(f"{target_alpha:12.1f} {knee_deg:10.2f} {alpha_actual:14.2f} {Fz:12.2f} {Fy:12.2f}")

# Check: are Rigney angles (12°, 22°) giving plausible forces?
print()
for ta in [12, 22]:
    if ta in fz_at_rigney:
        fz = fz_at_rigney[ta]
        print(f"  Fz at alpha={ta}°, Z=-10mm: {fz:.1f} N ")


# ============================================================
# Test 5c: Force-displacement curves at Rigney angles
# ============================================================

print("\n" + "="*50)
print("TEST 5c: Force-displacement curves at Rigney test angles")
print("="*50)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(f"ESR {ESR_MODEL_TYPE}: Force vs. Compression at Rigney angles (Rigney 2018)")

colors  = ['gray', 'blue', 'red']
labels  = ['alpha≈0°  (static)', 'alpha≈12° (Rigney)', 'alpha≈22° (Rigney)']
targets = [0, 12, 22]

for target_alpha, color, label in zip(targets, colors, labels):
    knee_deg = -(target_alpha + 7.66)

    # One mjx.forward to set geometry
    d_base = data.replace(
        qpos=data.qpos.at[knee_qpos].set(np.deg2rad(knee_deg))
    )
    d_base = mjx.forward(env.sys, d_base)

    alpha_actual = np.rad2deg(float(env.get_pylon_alpha(d_base, SIDE)))
    print(f"  target={target_alpha}° -> actual alpha={alpha_actual:.2f}°")

    fz_alpha = []
    fy_alpha = []

    # No mjx.forward needed inside loop - only qpos changes, geometry fixed
    for z_mm in z_sweep:
        d = d_base.replace(
            qpos=d_base.qpos.at[info["ty_qpos"]].set(float(z_mm) / 1000.0)
        )
        Fz, Fy = get_forces(d)
        fz_alpha.append(Fz)
        fy_alpha.append(Fy)

    axes2[0].plot(z_sweep, fz_alpha, color=color, linewidth=2, label=label)
    axes2[1].plot(z_sweep, fy_alpha, color=color, linewidth=2, label=label)

# Fz plot
axes2[0].axhline(y=1000, color='k', linestyle='--', linewidth=1.5,
                 label='Rigney max load (1000N)')
axes2[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes2[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes2[0].set_xlabel("Z displacement [mm]")
axes2[0].set_ylabel("Fz [N]")
axes2[0].set_title("Vertical force vs. compression")
axes2[0].legend()
axes2[0].grid(True)

# Fy plot
axes2[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes2[1].set_xlabel("Z displacement [mm]")
axes2[1].set_ylabel("Fy [N]")
axes2[1].set_title("Horizontal force vs. compression")
axes2[1].legend()
axes2[1].grid(True)

plt.tight_layout()
plt.savefig("esr_force_alpha_comparison.png", dpi=150)
print("  Plot saved: esr_force_alpha_comparison.png")
# ============================================================
# Test 6: Force-displacement plot
# ============================================================

print("\n" + "="*50)
print("TEST 6: Force-displacement curve (visual)")
print("="*50)

fz_sweep = []
fy_sweep = []

for z_mm in z_sweep:
    d = set_displacement(data, z_mm=float(z_mm))
    Fz, Fy = get_forces(d)
    fz_sweep.append(Fz)
    fy_sweep.append(Fy)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"ESR Force Model: {ESR_MODEL_TYPE} (Rigney 2018, Vari-Flex Modular)")

# Fz vs Z
axes[0].plot(z_sweep, fz_sweep, 'b-', linewidth=2)
axes[0].axhline(y=1000, color='r', linestyle='--', label='Rigney max load (1000N)')
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[0].set_xlabel("Z displacement [mm]")
axes[0].set_ylabel("Fz [N]")
axes[0].set_title("Vertical force vs. compression")
axes[0].legend()
axes[0].grid(True)

# Fy vs Z (should be 0 for linear elastic)
axes[1].plot(z_sweep, fy_sweep, 'g-', linewidth=2)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1].set_xlabel("Z displacement [mm]")
axes[1].set_ylabel("Fy [N]")
axes[1].set_title("Horizontal force vs. compression")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("esr_force_displacement.png", dpi=150)
print("  Plot saved: esr_force_displacement.png")

# Sanity check: force at max compression
fz_at_10mm = fz_sweep[0]  # z=-10mm
print(f"  Fz at Z=-10mm: {fz_at_10mm:.1f} N")

print("\n" + "="*50)
print("All tests complete.")
print("="*50)