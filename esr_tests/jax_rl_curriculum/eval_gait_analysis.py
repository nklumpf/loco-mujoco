"""
Gait-cycle analysis for ESR prosthesis evaluation.

This script processes joint angle, moment, and power data recorded during
MJX evaluation. The analysis is purely kinematic (no GRF/contact signals)
and uses knee-flexion minima to segment gait cycles for each leg.

Pipeline overview:
    1. Load raw joint data from CSV
    2. Detect gait cycles from knee-angle minima (left/right independently)
    3. Normalize each cycle to a 0-100% gait-cycle grid
    4. Apply sign conventions
    5. Mass-normalize moment/power (Nm/kg, W/kg)
    6. Load literature reference curves
    7. Compute phase alignment:
         - Knee: literature maximum knee flexion
         - Hip: knee shift + HIP_EXTRA_SHIFT
    8. Compute RMSE vs. literature
    9. Save RMSE results to rmse_result.csv
   10. Compute ROM statistics
   11. Plot ROM comparison and save rom_comparison.png
   12. Plot left/right knee and hip signals
   13. Optional: run diagnostics to evaluate best sign/shift settings
"""

# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks



# CONFIG
PROSTHESIS_SUBTYPE = "ESR"
ESR_MODEL_TYPE = "linear"
BODY_MASS_KG = 86.6
NORMALIZE_MOMENT_POWER_BY_MASS = True
# Gait-cycle detection
MIN_PEAK_DISTANCE = 80
MIN_PEAK_PROMINENCE = 5
# Remove first cycles because of initialization effects
N_WARMUP_CYCLES = 2
# Ignore final incomplete cycle
DROP_LAST_CYCLE = True
# Number of points for normalized gait cycle
N_POINTS = 200
# Additional hip phase shift in % gait cycle
HIP_EXTRA_SHIFT = 5
# Whether to run the sign/phase-shift diagnostic report
RUN_DIAGNOSTICS = True


def get_plot_dir(prosthesis_subtype, esr_model_type=None):
    base_dir = "plots"

    plot_dir = os.path.join(base_dir, prosthesis_subtype)

    if prosthesis_subtype == "ESR" and esr_model_type is not None:
        plot_dir = os.path.join(plot_dir, esr_model_type)

    os.makedirs(plot_dir, exist_ok=True)

    return plot_dir


# sign conventions
FLIP_SIGN_OVERRIDE = {
    "hip_angle_left_deg": False,
    "hip_angle_right_deg": False,
    "hip_power_left": False,
    "hip_power_right": False,
    "knee_moment_right": False,
    "knee_power_right": False,}

# Optional per-column phase-shift overrides
PHASE_SHIFT_PERCENT = {}


def should_flip(column_name: str) -> bool:
    """
    Determine whether a signal should be sign-flipped before
    cycle normalization.
    Rules:
        - Flip all knee angles unless overridden.
        - Flip all hip angles unless overridden.
        - Flip all hip kinetics unless overridden.
        - Do NOT flip knee moment/power unless overridden.
    Explicit entries in FLIP_SIGN_OVERRIDE take precedence.
    """

    if column_name in FLIP_SIGN_OVERRIDE:
        return FLIP_SIGN_OVERRIDE[column_name]

    name = column_name.lower()

    # Knee angles
    if "knee_angle" in name:
        return True

    # Hip angles
    if "hip_angle" in name:
        return True

    # Hip moment and power
    if "hip_moment" in name or "hip_power" in name:
        return True

    # Knee moment and power
    return False


def detect_knee_cycles(knee_angle):
    """
    Detect gait-cycle boundaries using maximum knee flexion.

    MuJoCo convention:
        Flexion = negative angle.

    Therefore:
        find_peaks(-knee_angle)
    identifies maximum knee flexion.

    Each leg is segmented independently.
    """
    peaks, _ = find_peaks(-knee_angle, distance=MIN_PEAK_DISTANCE, prominence=MIN_PEAK_PROMINENCE)

    return peaks


def trim_cycles(raw_starts, label=""):
    """
    Remove warm-up cycles and optionally the final incomplete cycle.

    cycle_starts[i] -> cycle_starts[i+1]
    defines one complete gait cycle.
    """
    if DROP_LAST_CYCLE:
        trimmed = raw_starts[N_WARMUP_CYCLES:-1]
    else:
        trimmed = raw_starts[N_WARMUP_CYCLES:]

    if len(trimmed) < 2:
        raise ValueError(f"Not enough gait cycles detected for '{label}'.")

    return trimmed


def normalize_cycle(signal, start, end):
    """
    Extract one gait cycle and resample it onto a uniform 0-100% gait-cycle grid.
    """
    cyc = signal[start:end]

    x_original = np.linspace(0, 100, len(cyc))
    x_norm = np.linspace(0, 100, N_POINTS) 

    return np.interp(x_norm, x_original, cyc)


def compute_mean_std(signal, cycle_starts, flip_sign=False):
    """
    Compute mean and standard deviation across all normalized gait cycles.
    Returns:
        mean_curve
        std_curve
    """
    signal = np.asarray(signal)
    if flip_sign:
        signal = -signal

    all_cycles = []

    for i in range(len(cycle_starts) - 1):
        cyc_norm = normalize_cycle(signal, cycle_starts[i], cycle_starts[i + 1])
        all_cycles.append(cyc_norm)

    all_cycles = np.vstack(all_cycles)
    mean_curve = np.mean(all_cycles, axis=0)
    std_curve = np.std(all_cycles, axis=0)

    return mean_curve, std_curve


def compute_rom_stats(signal, cycle_starts, flip_sign=False):
    """
    Compute ROM for every individual gait cycle.
    ROM = max(signal) - min(signal)
    Returns:
        mean_rom
        std_rom
        all_rom
    """
    signal = np.asarray(signal)

    if flip_sign:
        signal = -signal

    all_rom = []

    for i in range(len(cycle_starts) - 1):
        cyc_norm = normalize_cycle(signal, cycle_starts[i], cycle_starts[i + 1])
        cycle_rom = (np.max(cyc_norm) - np.min(cyc_norm))
        all_rom.append(cycle_rom)

    all_rom = np.asarray(all_rom)
    mean_rom = float(np.mean(all_rom))
    std_rom = float(np.std(all_rom))

    return mean_rom, std_rom, all_rom


# mass normalization
def per_kg(signal):
    return signal / BODY_MASS_KG


# phase alignment
def apply_phase_shift(curve, percent):
    """
    Circularly shift a normalized gait-cycle curve by a given percentage.
    """
    shift = int(round((percent / 100.0)* len(curve)))

    return np.roll(curve, shift)


def load_literature(path):
    """
    Load a literature reference curve from CSV.
    CSV:
        column 0 = gait-cycle percentage
        column 1 = signal value
    """
    df_lit = pd.read_csv(path, sep=",", header=None)

    gait = df_lit.iloc[:, 0].values
    vals = df_lit.iloc[:, 1].values

    idx = np.argsort(gait)

    return (gait[idx], vals[idx])


def get_literature_knee_peak(gait_lit, knee_angle_lit):
    """
    Identify gait-cycle percentage of maximum knee flexion in literature.
    """
    idx = np.argmax(knee_angle_lit)

    return (gait_lit[idx], knee_angle_lit[idx])


def interp_literature_to_xnorm(gait_lit, lit_vals, x_norm):
    return np.interp(x_norm, gait_lit, lit_vals)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def find_best_flip_and_shift(sim_curve, lit_curve_on_grid):
    """
    Brute-force search over all circular shifts and both sign options.
    This is only diagnostic and does not modify the manually selected configuration.
    """
    n = len(sim_curve)

    baseline_rmse = rmse(sim_curve, lit_curve_on_grid)

    best_shift = 0
    best_flip = False
    best_rmse = baseline_rmse

    for flip in (False, True):
        curve = (
            -sim_curve
            if flip
            else sim_curve)

        for shift in range(n):
            shifted = np.roll(curve, shift)
            err = rmse(shifted, lit_curve_on_grid)

            if err < best_rmse:
                best_shift = shift
                best_flip = flip
                best_rmse = err

    return {
        "flip": best_flip,
        "shift_percent":
            100.0 * best_shift / n,
        "rmse": best_rmse,
        "baseline_rmse":
            baseline_rmse,
        "improvement":
            baseline_rmse - best_rmse,}


def evaluate_flip_at_fixed_shift(sim_curve, lit_curve_on_grid, shift_percent):
    """
    Test flip=True/False at a fixed phase shift.
    """
    n = len(sim_curve)

    shift = int(round((shift_percent / 100.0) * n))

    results = {}

    for flip in (False, True):
        curve = (
            -sim_curve
            if flip
            else sim_curve)
        shifted = np.roll(curve, shift)
        results[flip] = rmse(shifted, lit_curve_on_grid)

    best_flip = min(results, key=results.get)

    return {
        "flip_false_rmse":
            results[False],
        "flip_true_rmse":
            results[True],
        "recommended_flip":
            best_flip,}


def run_sign_shift_diagnostics(df, cycle_left, cycle_right, x_norm, literature_curves, reference_shift_percent=None, knee_shift_percent=None, hip_shift_percent=None,):
    """
    Run diagnostic search for all 12 standard signals.
    """
    signal_specs = [
        ("Knee angle", "knee_angle_left_deg", "knee_angle_right_deg", "knee_angle", False),
        ("Knee moment", "knee_moment_left", "knee_moment_right", "knee_moment", True),
        ("Knee power", "knee_power_left", "knee_power_right", "knee_power", True),
        ("Hip angle", "hip_angle_left_deg", "hip_angle_right_deg", "hip_angle", False),
        ("Hip moment", "hip_moment_left", "hip_moment_right", "hip_moment", True),
        ("Hip power", "hip_power_left", "hip_power_right", "hip_power", True),]

    print("\n" + "=" * 100)
    print("SIGN / PHASE-SHIFT DIAGNOSTICS "
          "(search result -- verify before adopting!)")

    if reference_shift_percent is not None:
        print(
            f"Trusted reference shift "
            f"(knee-flexion peak): "
            f"{reference_shift_percent:.1f}%")

    print("=" * 100)

    header = (
        f"{'signal':14s} "
        f"{'side':6s} "
        f"{'flip?':6s} "
        f"{'shift%':8s} "
        f"{'delta_ref':10s} "
        f"{'RMSE':8s} "
        f"{'baseline':9s} "
        f"{'improved':9s} "
        f"{'@ref:flip':10s} "
        f"{'@ref:RMSE':9s}")

    print(header)
    print("-" * len(header))

    for (label, col_left, col_right, lit_key, mass_norm) in signal_specs:

        gait_lit, lit_vals = literature_curves[lit_key]
        lit_on_grid = interp_literature_to_xnorm(gait_lit, lit_vals, x_norm)

        ref_shift = (
            knee_shift_percent
            if "Knee" in label
            else hip_shift_percent)

        for side, col, cycle_starts in [
            ("left", col_left, cycle_left),
            ("right", col_right, cycle_right),]:

            sig = df[col].values

            if mass_norm:
                sig = per_kg(sig)

            mean_raw, _ = compute_mean_std(sig, cycle_starts, flip_sign=False)

            result = find_best_flip_and_shift(mean_raw, lit_on_grid)

            if reference_shift_percent is not None:
                delta = (result["shift_percent"] - reference_shift_percent)
                delta_str = f"{delta:+7.1f}%"

            else:
                delta_str = "n/a"

            if ref_shift is not None:
                fixed = evaluate_flip_at_fixed_shift(mean_raw, lit_on_grid, ref_shift)
                at_ref_flip_str = str(fixed["recommended_flip"])
                if fixed["recommended_flip"]:
                    at_ref_rmse = fixed["flip_true_rmse"]
                else:
                    at_ref_rmse = fixed["flip_false_rmse"]

                at_ref_rmse_str = f"{at_ref_rmse:.3f}"

            else:
                at_ref_flip_str = "n/a"
                at_ref_rmse_str = "n/a"

            print(
                f"{label:14s} "
                f"{side:6s} "
                f"{str(result['flip']):6s} "
                f"{result['shift_percent']:7.1f}% "
                f"{delta_str:10s} "
                f"{result['rmse']:8.3f} "
                f"{result['baseline_rmse']:9.3f} "
                f"{result['improvement']:9.3f} "
                f"{at_ref_flip_str:10s} "
                f"{at_ref_rmse_str:9s}")


def plot_into_axis(ax, x_norm, mean, std, gait_lit, lit_vals, title, unit=""):
    ax.plot(x_norm, mean, linewidth=2, label="Simulation mean")
    ax.fill_between(x_norm, mean - std, mean + std, alpha=0.2, label="Simulation +/-1 SD")
    ax.plot(gait_lit, lit_vals, linestyle="--", linewidth=2, label="Literature")

    ax.set_title(title)
    ax.set_xlabel("Gait Cycle [%]")
    ax.set_ylabel(f"Value" f"{f' [{unit}]' if unit else ''}")
    ax.grid(True)
    ax.legend(fontsize=8)


# ============================================================
# MAIN
def main():
    # Plot directory
    plot_dir = get_plot_dir(PROSTHESIS_SUBTYPE, ESR_MODEL_TYPE)

    # --------------------------------------------------------
    # Load raw simulation data
    raw_data_path = os.path.join(plot_dir, "raw_joint_data.csv")
    df = pd.read_csv(raw_data_path)
    x_norm = np.linspace(0, 100, N_POINTS)

    # --------------------------------------------------------
    # Load literature curves
    gait_knee_angle, lit_knee_angle   = load_literature("../gait-data/ESR_knee_angle.csv")
    gait_knee_moment, lit_knee_moment = load_literature("../gait-data/ESR_knee_moment.csv")
    gait_knee_power, lit_knee_power   = load_literature("../gait-data/ESR_knee_power.csv")
    gait_hip_angle, lit_hip_angle     = load_literature("../gait-data/ESR_hip_angle.csv")
    gait_hip_moment, lit_hip_moment   = load_literature("../gait-data/ESR_hip_moment.csv")
    gait_hip_power, lit_hip_power     = load_literature("../gait-data/ESR_hip_power.csv")

    literature_curves = {
        "knee_angle": (gait_knee_angle, lit_knee_angle),
        "knee_moment": (gait_knee_moment, lit_knee_moment),
        "knee_power": (gait_knee_power, lit_knee_power),
        "hip_angle": (gait_hip_angle, lit_hip_angle),
        "hip_moment": (gait_hip_moment, lit_hip_moment),
        "hip_power": (gait_hip_power, lit_hip_power),}

    # --------------------------------------------------------
    # Determine phase shifts
    knee_shift, _ = get_literature_knee_peak(gait_knee_angle, lit_knee_angle)
    hip_shift = (knee_shift + HIP_EXTRA_SHIFT)

    PHASE_SHIFT = {
        "knee":
            knee_shift,
        "hip":
            hip_shift,}

    print(f"Knee phase shift: " f"{knee_shift:.2f}%")
    print(f"Hip phase shift:  " f"{hip_shift:.2f}%")

    # Detect gait cycles
    raw_left    = detect_knee_cycles(df["knee_angle_left_deg"].values)
    raw_right   = detect_knee_cycles(df["knee_angle_right_deg"].values)
    cycle_left  = trim_cycles(raw_left, label="knee_left")
    cycle_right = trim_cycles(raw_right, label="knee_right")

    print(f"\nDetected usable left cycles: "
          f"{len(cycle_left) - 1}")
    print(f"Detected usable right cycles: "
          f"{len(cycle_right) - 1}")

    # Sign / phase-shift diagnostics
    if RUN_DIAGNOSTICS:
        run_sign_shift_diagnostics(
            df,
            cycle_left,
            cycle_right,
            x_norm,
            literature_curves,
            reference_shift_percent=knee_shift,
            knee_shift_percent=knee_shift,
            hip_shift_percent=hip_shift)

    # ========================================================

    def joint_stats(col_left, col_right, mass_norm=False):
        """
        Compute mean/std curves for left and right signals.

        Steps:
            1. Extract raw signals.
            2. Optionally mass-normalize.
            3. Apply sign convention.
            4. Normalize each gait cycle.
            5. Apply phase shift.
            6. Return mean/std curves.
        """
        sig_left = df[col_left].values
        sig_right = df[col_right].values

        if mass_norm:
            sig_left = per_kg(sig_left)
            sig_right = per_kg(sig_right)

        mean_l, std_l = compute_mean_std(sig_left, cycle_left, flip_sign=should_flip(col_left))
        mean_r, std_r = compute_mean_std(sig_right, cycle_right, flip_sign=should_flip(col_right))

        # Determine joint-level default shift
        if "knee" in col_left:
            default_shift = PHASE_SHIFT["knee"]
        else:
            default_shift = PHASE_SHIFT["hip"]

        shift_l = PHASE_SHIFT_PERCENT.get(col_left, default_shift)
        shift_r = PHASE_SHIFT_PERCENT.get(col_right, default_shift)

        mean_l = apply_phase_shift(mean_l, shift_l)
        mean_r = apply_phase_shift(mean_r, shift_r)

        std_l = apply_phase_shift(std_l, shift_l)
        std_r = apply_phase_shift(std_r, shift_r)

        return (mean_l, std_l, mean_r, std_r)

    # compute knee signals 
    (knee_angle_left_mean, knee_angle_left_std, knee_angle_right_mean, knee_angle_right_std) = joint_stats("knee_angle_left_deg", "knee_angle_right_deg")
    (knee_moment_left_mean, knee_moment_left_std, knee_moment_right_mean, knee_moment_right_std) = joint_stats("knee_moment_left", "knee_moment_right", mass_norm=True)
    (knee_power_left_mean, knee_power_left_std, knee_power_right_mean, knee_power_right_std) = joint_stats("knee_power_left", "knee_power_right", mass_norm=True)

    # compute hip signals
    (hip_angle_left_mean, hip_angle_left_std, hip_angle_right_mean, hip_angle_right_std) = joint_stats("hip_angle_left_deg", "hip_angle_right_deg")
    (hip_moment_left_mean, hip_moment_left_std, hip_moment_right_mean, hip_moment_right_std) = joint_stats("hip_moment_left", "hip_moment_right", mass_norm=True)
    (hip_power_left_mean, hip_power_left_std, hip_power_right_mean, hip_power_right_std) = joint_stats("hip_power_left", "hip_power_right", mass_norm=True)

    # RMSE
    rmse_results = []

    def report_rmse(name, curve, gait_lit, lit_vals):
        """
        Compute RMSE, print it, and store it for CSV export.
        """
        lit_on_grid = interp_literature_to_xnorm(gait_lit, lit_vals, x_norm)

        value = rmse(curve, lit_on_grid)

        print(f"{name:22s} "
              f"RMSE: {value:.3f}")

        rmse_results.append({"Signal": name, "RMSE": value})

    print("\nRMSE vs literature " "(using should_flip() / " "PHASE_SHIFT config above):")

    report_rmse("Knee angle L", knee_angle_left_mean, gait_knee_angle, lit_knee_angle)
    report_rmse("Knee angle R", knee_angle_right_mean, gait_knee_angle, lit_knee_angle)
    report_rmse("Knee moment L", knee_moment_left_mean, gait_knee_moment, lit_knee_moment)
    report_rmse("Knee moment R", knee_moment_right_mean, gait_knee_moment, lit_knee_moment)
    report_rmse("Knee power L", knee_power_left_mean, gait_knee_power, lit_knee_power)
    report_rmse("Knee power R", knee_power_right_mean, gait_knee_power, lit_knee_power)

    report_rmse("Hip angle L", hip_angle_left_mean, gait_hip_angle, lit_hip_angle)
    report_rmse("Hip angle R", hip_angle_right_mean, gait_hip_angle, lit_hip_angle)
    report_rmse("Hip moment L", hip_moment_left_mean, gait_hip_moment, lit_hip_moment)
    report_rmse("Hip moment R", hip_moment_right_mean, gait_hip_moment, lit_hip_moment)
    report_rmse("Hip power L", hip_power_left_mean, gait_hip_power, lit_hip_power)
    report_rmse("Hip power R", hip_power_right_mean, gait_hip_power, lit_hip_power)

    # Save RMSE CSV
    rmse_df = pd.DataFrame(rmse_results)
    rmse_path = os.path.join(plot_dir, "rmse_result.csv")
    rmse_df.to_csv(rmse_path, index=False)
    print(f"\nRMSE results saved to:")
    print(rmse_path)

    # ========================================================
    # ROM ANALYSIS
    print("\n" + "=" * 70)
    print("RANGE OF MOTION (ROM)")
    print("=" * 70)

    # Simulation ROM
    (knee_rom_left_mean, knee_rom_left_std, knee_rom_left_all)    = compute_rom_stats(df["knee_angle_left_deg"].values, cycle_left, flip_sign=should_flip("knee_angle_left_deg"))
    (knee_rom_right_mean, knee_rom_right_std, knee_rom_right_all) = compute_rom_stats(df["knee_angle_right_deg"].values, cycle_right, flip_sign=should_flip("knee_angle_right_deg"))

    (hip_rom_left_mean, hip_rom_left_std, hip_rom_left_all)    = compute_rom_stats(df["hip_angle_left_deg"].values, cycle_left, flip_sign=should_flip("hip_angle_left_deg"))
    (hip_rom_right_mean, hip_rom_right_std, hip_rom_right_all) = compute_rom_stats(df["hip_angle_right_deg"].values, cycle_right, flip_sign=should_flip("hip_angle_right_deg"))

    # Literature ROM
    lit_knee_rom = (np.max(lit_knee_angle) - np.min(lit_knee_angle))
    lit_hip_rom  = (np.max(lit_hip_angle) - np.min(lit_hip_angle))

    print(
        f"Knee ROM Left:  "
        f"Simulation = "
        f"{knee_rom_left_mean:.2f} "
        f"+/- {knee_rom_left_std:.2f} deg, "
        f"Literature = "
        f"{lit_knee_rom:.2f} deg")

    print(
        f"Knee ROM Right: "
        f"Simulation = "
        f"{knee_rom_right_mean:.2f} "
        f"+/- {knee_rom_right_std:.2f} deg, "
        f"Literature = "
        f"{lit_knee_rom:.2f} deg")

    print(
        f"Hip ROM Left:   "
        f"Simulation = "
        f"{hip_rom_left_mean:.2f} "
        f"+/- {hip_rom_left_std:.2f} deg, "
        f"Literature = "
        f"{lit_hip_rom:.2f} deg")

    print(
        f"Hip ROM Right:  "
        f"Simulation = "
        f"{hip_rom_right_mean:.2f} "
        f"+/- {hip_rom_right_std:.2f} deg, "
        f"Literature = "
        f"{lit_hip_rom:.2f} deg")

    # ========================================================
    # ROM COMPARISON PLOT
    rom_labels = [
        "Knee ROM Left",
        "Knee ROM Right",
        "Hip ROM Left",
        "Hip ROM Right"]

    literature_rom = np.array([
        lit_knee_rom,
        lit_knee_rom,
        lit_hip_rom,
        lit_hip_rom])

    simulation_rom = np.array([
        knee_rom_left_mean,
        knee_rom_right_mean,
        hip_rom_left_mean,
        hip_rom_right_mean])

    simulation_rom_std = np.array([
        knee_rom_left_std,
        knee_rom_right_std,
        hip_rom_left_std,
        hip_rom_right_std])

    x = np.arange(len(rom_labels))

    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    # Literature bars
    ax.bar(
        x - width / 2,
        literature_rom,
        width,
        label="Literature")

    # Simulation bars with SD
    ax.bar(
        x + width / 2,
        simulation_rom,
        width,
        yerr=simulation_rom_std,
        capsize=5,
        label="Simulation")
    ax.set_ylabel("ROM [deg]")
    ax.set_title("Range of Motion: "
                 "Simulation vs Literature")
    ax.set_xticks(x)
    ax.set_xticklabels(rom_labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    rom_plot_path = os.path.join(plot_dir, "rom_comparison.png")
    plt.savefig(rom_plot_path, dpi=150)
    plt.show()
    print(f"\nROM comparison plot saved to:")
    print(rom_plot_path)

    # ========================================================
    # knee plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        "Knee Joint Comparison - "
        "Left vs Right "
        "(Mean +/- SD vs Literature)",
        fontsize=16)

    plot_into_axis(
        axes[0, 0],
        x_norm,
        knee_angle_left_mean,
        knee_angle_left_std,
        gait_knee_angle,
        lit_knee_angle,
        "Knee Angle Left",
        "deg")

    plot_into_axis(
        axes[0, 1],
        x_norm,
        knee_angle_right_mean,
        knee_angle_right_std,
        gait_knee_angle,
        lit_knee_angle,
        "Knee Angle Right",
        "deg")

    plot_into_axis(
        axes[1, 0],
        x_norm,
        knee_moment_left_mean,
        knee_moment_left_std,
        gait_knee_moment,
        lit_knee_moment,
        "Knee Moment Left",
        "Nm/kg")

    plot_into_axis(
        axes[1, 1],
        x_norm,
        knee_moment_right_mean,
        knee_moment_right_std,
        gait_knee_moment,
        lit_knee_moment,
        "Knee Moment Right",
        "Nm/kg")

    plot_into_axis(
        axes[2, 0],
        x_norm,
        knee_power_left_mean,
        knee_power_left_std,
        gait_knee_power,
        lit_knee_power,
        "Knee Power Left",
        "W/kg")

    plot_into_axis(
        axes[2, 1],
        x_norm,
        knee_power_right_mean,
        knee_power_right_std,
        gait_knee_power,
        lit_knee_power,
        "Knee Power Right",
        "W/kg")

    plt.tight_layout()
    knee_plot_path = os.path.join(plot_dir, "knee_subplot_6panel.png")
    plt.savefig(knee_plot_path, dpi=150)
    plt.show()

    # ========================================================
    # hip plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        "Hip Joint Comparison - "
        "Left vs Right "
        "(Mean +/- SD vs Literature)",
        fontsize=16)

    plot_into_axis(
        axes[0, 0],
        x_norm,
        hip_angle_left_mean,
        hip_angle_left_std,
        gait_hip_angle,
        lit_hip_angle,
        "Hip Angle Left",
        "deg")

    plot_into_axis(
        axes[0, 1],
        x_norm,
        hip_angle_right_mean,
        hip_angle_right_std,
        gait_hip_angle,
        lit_hip_angle,
        "Hip Angle Right",
        "deg")

    plot_into_axis(
        axes[1, 0],
        x_norm,
        hip_moment_left_mean,
        hip_moment_left_std,
        gait_hip_moment,
        lit_hip_moment,
        "Hip Moment Left",
        "Nm/kg")

    plot_into_axis(
        axes[1, 1],
        x_norm,
        hip_moment_right_mean,
        hip_moment_right_std,
        gait_hip_moment,
        lit_hip_moment,
        "Hip Moment Right",
        "Nm/kg")

    plot_into_axis(
        axes[2, 0],
        x_norm,
        hip_power_left_mean,
        hip_power_left_std,
        gait_hip_power,
        lit_hip_power,
        "Hip Power Left",
        "W/kg")

    plot_into_axis(
        axes[2, 1],
        x_norm,
        hip_power_right_mean,
        hip_power_right_std,
        gait_hip_power,
        lit_hip_power,
        "Hip Power Right",
        "W/kg")

    plt.tight_layout()
    hip_plot_path = os.path.join(plot_dir, "hip_subplot_6panel.png")
    plt.savefig(hip_plot_path, dpi=150)
    plt.show()


# ============================================================
# RUN
if __name__ == "__main__":
    main()