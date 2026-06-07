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

print("\n" + "="*50)
print("TEST 5: Alpha plausibility")
print("="*50)
print("Expected: alpha in range [-25°, 25°] during normal stance")

alpha_rad = float(env.get_pylon_alpha(data, SIDE))
alpha_deg = np.rad2deg(alpha_rad)
in_range = -25.0 <= alpha_deg <= 25.0
result = PASS if in_range else FAIL
print(f"  alpha = {alpha_deg:.2f} deg  {result}")
print(f"  (Rigney tested at 12° and 22°)")


# ============================================================
# Test 6: Force-displacement plot
# ============================================================

print("\n" + "="*50)
print("TEST 6: Force-displacement curve (visual)")
print("="*50)

z_sweep = np.linspace(-30, 0, 100)
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
fz_at_30mm = fz_sweep[0]  # z=-30mm
print(f"  Fz at Z=-30mm: {fz_at_30mm:.1f} N")
print(f"  Rigney max load: 1000 N")
in_range = 500 < fz_at_30mm < 2000
result = PASS if in_range else FAIL
print(f"  Force in plausible range (500-2000N): {result}")

print("\n" + "="*50)
print("All tests complete.")
print("="*50)