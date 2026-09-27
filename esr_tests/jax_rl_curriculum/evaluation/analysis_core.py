"""
Core utilities for gait-cycle processing and literature comparison.

Includes cycle detection, normalization, mean/std curve computation, ROM statistics, sign conventions, phase-shift handling, mass normalization, and literature loading/interpolation. These functions form the backbone of joint-angle, moment, power, and GRF analysis across ESR and SACH prosthesis simulations.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

# CONFIG
BODY_MASS_KG = 86.6
NORMALIZE_MOMENT_POWER_BY_MASS = True
# Gait-cycle detection
MIN_PEAK_DISTANCE = 80
MIN_PEAK_PROMINENCE = 5
N_WARMUP_CYCLES = 2
DROP_LAST_CYCLE = True
N_POINTS = 200
# Hip phase shift
HIP_EXTRA_SHIFT = 0
# Diagnostics
RUN_DIAGNOSTICS = True
# GRF
GRF_THRESHOLD_N = 50.0

# Sign conventions
# NOTE: keys updated to match the UNIFIED column naming produced by the
# rewritten eval script: "{joint}_{side}_{quantity}" (e.g. "hip_left_angle_deg")
FLIP_SIGN_OVERRIDE = {
    "hip_left_angle_deg": False,
    "hip_right_angle_deg": False,
    "hip_left_power": False,
    "hip_right_power": False,
    "knee_right_moment": False,
    "knee_right_power": False}
# Optional per-column phase-shift overrides
PHASE_SHIFT_PERCENT = {}
# Will be set in main
PHASE_SHIFT = {"knee": 0.0, "hip": 0.0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gait cycle analysis for ESR/SACH prosthesis evaluation.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Direct path to raw_joint_data.csv. Overrides --subtype/--model-type/--timestamp if given.")
    parser.add_argument(
        "--subtype",
        type=str,
        default="ESR",
        help="Prosthesis subtype folder name, e.g. ESR or SACH.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="linear",
        help="ESR model type subfolder (linear/nonlinear). Ignored for SACH.")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Run timestamp subfolder, e.g. '2026-09-12_07-15-31'. If omitted, the most recent timestamped subfolder is used.")
    parser.add_argument("--plots", action="store_true", help="Show debug plots")


    return parser.parse_args()


def find_latest_run_dir(base_plot_dir):
    """
    Auto-detect the most recent run_timestamp subfolder (format YYYY-MM-DD_HH-MM-SS) under base_plot_dir. Relies on the timestamp string sorting chronologically, which holds for this exact format.
    """
    if not os.path.isdir(base_plot_dir):
        raise FileNotFoundError(f"Plot base directory not found: {base_plot_dir}")

    candidates = [
        d for d in os.listdir(base_plot_dir)
        if os.path.isdir(os.path.join(base_plot_dir, d))
        and re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d)]
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped run subfolders found under {base_plot_dir}. Pass --timestamp explicitly.")
    candidates.sort()
    return candidates[-1]


def get_plot_dir(prosthesis_subtype, esr_model_type=None, run_timestamp=None):
    base_dir = "plots"
    plot_dir = os.path.join(base_dir, prosthesis_subtype)

    if prosthesis_subtype == "ESR" and esr_model_type is not None:
        plot_dir = os.path.join(plot_dir, esr_model_type)
    if run_timestamp is not None:
        plot_dir = os.path.join(plot_dir, run_timestamp)
    os.makedirs(plot_dir, exist_ok=True)

    return plot_dir


def should_flip(column_name: str) -> bool:
    """
    Determine whether a signal should be sign-flipped before
    cycle normalization.

    Column names follow the UNIFIED pattern "{joint}_{side}_{quantity}",
    e.g. "ankle_left_moment", "knee_right_angle_deg", "hip_left_power".
    Matching is therefore done on (a) which joint the column starts with
    and (b) which quantity substring it contains, rather than on the old
    fixed substrings like "knee_angle" (which no longer appear verbatim
    once "angle" and "left"/"right" swapped position in the name).

    Rules:
        - Flip ankle moments unless overridden.
        - Flip all knee angles unless overridden.
        - Flip all hip angles unless overridden.
        - Flip all hip kinetics (moment/power) unless overridden.
        - Do NOT flip knee moment/power unless overridden.
        - Do NOT flip ankle angle/power unless overridden.
    Explicit entries in FLIP_SIGN_OVERRIDE take precedence.
    """
    if column_name in FLIP_SIGN_OVERRIDE:
        return FLIP_SIGN_OVERRIDE[column_name]

    name = column_name.lower()

    is_ankle_right = name.startswith("ankle_right")
    is_ankle_left = name.startswith("ankle_left")
    is_knee = name.startswith("knee")
    is_hip = name.startswith("hip")

    is_angle = "angle" in name
    is_moment = "moment" in name
    is_power = "power" in name

    if is_ankle_right:
        return True
    if is_ankle_left and is_moment:
        return True
    if is_knee and is_angle:
        return True
    if is_hip and is_angle:
        return True
    if is_hip and (is_moment or is_power):
        return True
    return False


def detect_knee_cycles(knee_angle):
    """
    Detect gait-cycle boundaries using maximum knee flexion.
    MuJoCo convention:
        Flexion = negative angle.
    Therefore:
        find_peaks(-knee_angle) identifies maximum knee flexion.
    Each leg is segmented independently.
    """
    peaks, _ = find_peaks(-knee_angle, distance=MIN_PEAK_DISTANCE, prominence=MIN_PEAK_PROMINENCE)

    return peaks


def trim_cycles(raw_starts, label=""):
    """
    Remove warm-up cycles and optionally the final incomplete cycle.
    cycle_starts[i] -> cycle_starts[i+1] defines one complete gait cycle.
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
    std_curve  = np.std(all_cycles, axis=0)

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
        cycle_rom = np.max(cyc_norm) - np.min(cyc_norm)
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
    shift = int(round((percent / 100.0) * len(curve)))
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

    return gait[idx], vals[idx]


def get_literature_knee_peak(gait_lit, knee_angle_lit):
    """
    Identify gait-cycle percentage of maximum knee flexion in literature.
    """
    idx = np.argmax(knee_angle_lit)

    return gait_lit[idx], knee_angle_lit[idx]


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
        curve = -sim_curve if flip else sim_curve
        for shift in range(n):
            shifted = np.roll(curve, shift)
            err = rmse(shifted, lit_curve_on_grid)
            if err < best_rmse:
                best_shift = shift
                best_flip = flip
                best_rmse = err

    return {
        "flip": best_flip,
        "shift_percent": 100.0 * best_shift / n,
        "rmse": best_rmse,
        "baseline_rmse": baseline_rmse,
        "improvement": baseline_rmse - best_rmse}


def evaluate_flip_at_fixed_shift(sim_curve, lit_curve_on_grid, shift_percent):
    """
    Test flip=True/False at a fixed phase shift.
    """
    n = len(sim_curve)
    shift = int(round((shift_percent / 100.0) * n))

    results = {}
    for flip in (False, True):
        curve = -sim_curve if flip else sim_curve
        shifted = np.roll(curve, shift)
        results[flip] = rmse(shifted, lit_curve_on_grid)

    best_flip = min(results, key=results.get)
    return {
        "flip_false_rmse": results[False],
        "flip_true_rmse": results[True],
        "recommended_flip": best_flip}


def run_sign_shift_diagnostics(df, cycle_left, cycle_right, x_norm, literature_curves):
    """
    Run diagnostic search for all standard signals.

    NOTE: col_left / col_right below use the UNIFIED column naming from the
    rewritten eval script ("{joint}_{side}_{quantity}"). The lit_key values
    are unchanged -- they only index into the literature_curves dict built
    in eval_analysis.py, which was not renamed.
    """
    signal_specs = [
        ("Ankle angle", "ankle_left_angle_deg", "ankle_right_angle_deg", "ankle_angle",  False),
        ("Ankle moment","ankle_left_moment",    "ankle_right_moment",    "ankle_moment", True),
        ("Ankle power", "ankle_left_power",     "ankle_right_power",     "ankle_power",  False),
        ("Knee angle",  "knee_left_angle_deg",  "knee_right_angle_deg",  "knee_angle",   False),
        ("Knee moment", "knee_left_moment",     "knee_right_moment",     "knee_moment",  True),
        ("Knee power",  "knee_left_power",      "knee_right_power",      "knee_power",   True),
        ("Hip angle",   "hip_left_angle_deg",   "hip_right_angle_deg",   "hip_angle",    False),
        ("Hip moment",  "hip_left_moment",      "hip_right_moment",      "hip_moment",   True),
        ("Hip power",   "hip_left_power",       "hip_right_power",       "hip_power",    True)]

    print("\n" + "=" * 100)
    print("SIGN / PHASE-SHIFT DIAGNOSTICS (search result -- verify before adopting!)")
    print("=" * 100)

    header = (
        f"{'signal':14s} {'side':6s} {'flip?':6s} {'shift%':8s} "
        f"{'RMSE':8s} {'baseline':9s} {'improved':9s} "
        f"{'@ref:flip':10s} {'@ref:RMSE':9s}")
    print(header)
    print("-" * len(header))

    for (label, col_left, col_right, lit_key, mass_norm) in signal_specs:
        for side, col, cycle_starts in [
            ("left",  col_left,  cycle_left),
            ("right", col_right, cycle_right)]:

            # Select correct literature depending on side
            lit_key_full = ("sound_" if side == "left" else "esr_") + lit_key
            gait_lit, lit_vals = literature_curves[lit_key_full]
            lit_on_grid = interp_literature_to_xnorm(gait_lit, lit_vals, x_norm)

            # Select correct reference shift depending on side + joint
            if side == "left":
                ref_shift = PHASE_SHIFT["knee_l"]
            else:
                ref_shift = PHASE_SHIFT["knee_r"]

            sig = df[col].values
            if mass_norm:
                sig = per_kg(sig)

            mean_raw, _ = compute_mean_std(sig, cycle_starts, flip_sign=False)
            result = find_best_flip_and_shift(mean_raw, lit_on_grid)

            fixed = evaluate_flip_at_fixed_shift(mean_raw, lit_on_grid, ref_shift)
            at_ref_flip_str = str(fixed["recommended_flip"])
            at_ref_rmse = fixed["flip_true_rmse"] if fixed["recommended_flip"] else fixed["flip_false_rmse"]

            print(
                f"{label:14s} {side:6s} {str(result['flip']):6s} "
                f"{result['shift_percent']:7.1f}% "
                f"{result['rmse']:8.3f} {result['baseline_rmse']:9.3f} "
                f"{result['improvement']:9.3f} "
                f"{at_ref_flip_str:10s} {at_ref_rmse:.3f}")



def plot_into_axis(ax, x_norm, mean, std, gait_lit=None, lit_vals=None, title="", unit=""):
    ax.plot(x_norm, mean, linewidth=2, label="Simulation mean")
    ax.fill_between(x_norm, mean - std, mean + std, alpha=0.2, label="Simulation +/-1 SD")
    if gait_lit is not None and lit_vals is not None:
        ax.plot(gait_lit, lit_vals, linestyle="--", linewidth=2, label="Literature")

    ax.set_title(title)
    ax.set_xlabel("Gait Cycle [%]")
    ax.set_ylabel(f"Value{f' [{unit}]' if unit else ''}")
    ax.grid(True)
    ax.legend(fontsize=8)


def joint_stats(df, cycle_left, cycle_right, col_left, col_right, mass_norm=False):
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
    def process_side(col, cycle, suffix):
        if col is None:
            return None, None

        sig = df[col].values
        if mass_norm:
            sig = per_kg(sig)

        mean, std = compute_mean_std(sig, cycle, flip_sign=should_flip(col))
        shift = PHASE_SHIFT["knee_" + suffix if "knee" in col else "hip_" + suffix]

        return apply_phase_shift(mean, shift), apply_phase_shift(std, shift)

    mean_l, std_l = process_side(col_left, cycle_left, "l")
    mean_r, std_r = process_side(col_right, cycle_right, "r")

    return mean_l, std_l, mean_r, std_r