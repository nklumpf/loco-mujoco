"""
Symmetry analysis module for ESR/SACH gait evaluation.

Provides GRF smoothing, contact detection, gait-event extraction, stance/swing estimation, step-length computation, and left/right curve symmetry metrics (RMSE, magnitude RMSE, normalized difference).
Also includes temporal symmetry, ROM symmetry, and GRF-based gait phase analysis. Used by main_eval to generate per-run symmetry reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from analysis_core import (
    parse_args,
    rmse,
    compute_mean_std,
    compute_rom_stats,
    should_flip,
    apply_phase_shift,
    per_kg,
    PHASE_SHIFT,
    PHASE_SHIFT_PERCENT)

GRF_THRESHOLD = 150


def compute_grf_threshold(body_mass_kg, bw_fraction=0.05):
    """
    Compute GRF threshold as body-weight fraction.
    """
    return bw_fraction * body_mass_kg * 9.81


def smooth_grf(grf_raw):
    """
    Strong GRF smoothing + noise clamp for ESR signals.
    """
    grf_smooth = savgol_filter(grf_raw, 27, 3)
    grf_smooth[grf_smooth < 100] = 0.0
    return grf_smooth


def detect_contact(grf_smooth, threshold):
    """
    Simple contact mask: GRF above threshold.
    """
    return grf_smooth > threshold

def detect_contact_hysteresis(grf_smooth, enter_threshold, exit_threshold):
    """
    Hysteresis contact mask to avoid threshold flicker.
    """
    contact = np.zeros(len(grf_smooth), dtype=bool)
    in_contact = False
    for i, v in enumerate(grf_smooth):
        if not in_contact and v > enter_threshold:
            in_contact = True
        elif in_contact and v < exit_threshold:
            in_contact = False
        contact[i] = in_contact
    return contact


def detect_events(contact):
    """
    Detect heel-strike (HS) and toe-off (TO) from contact transitions.
    """
    hs = np.where(contact[1:] & ~contact[:-1])[0] + 1
    to = np.where(~contact[1:] & contact[:-1])[0] + 1
    return hs, to

def detect_events_kinematic(foot_pos, pelvis_pos, axis=0):
    """
    Kinematic HS/TO detection via foot‑pelvis relative motion.
    """
    rel = foot_pos[:, axis] - pelvis_pos[:, axis]
    hs, _ = find_peaks(rel, distance=20)
    to, _ = find_peaks(-rel, distance=20)
    return hs, to


def trim_cycles(hs, to):
    """
    Remove first two and last gait cycle for stability.
    """
    if len(hs) < 4:
        print("WARNING: Not enough cycles to trim.")
        return hs, to

    hs_trim = hs[2:-1]
    to_trim = to[2:-1]

    return hs_trim, to_trim


def compute_stance_swing_percent(hs, to):
    """
    Compute stance/swing percentage per gait cycle.
    """
    stance_pct = []
    swing_pct = []

    for i in range(len(hs) - 1):
        cycle_start = hs[i]
        cycle_end   = hs[i + 1]
        toe_off_candidates = to[(to > cycle_start) & (to < cycle_end)]
        if len(toe_off_candidates) == 0:
            continue
        toe_off = toe_off_candidates[0]

        cycle_len  = cycle_end - cycle_start
        stance_len = toe_off - cycle_start

        stance_pct.append(100.0 * stance_len / cycle_len)
        swing_pct.append(100.0 - stance_pct[-1])

    return np.array(stance_pct), np.array(swing_pct)


def compute_step_length(foot_L, foot_R, hs_L, hs_R, axis=0):
    """
    Compute left/right step length using HS positions.
    """
    # Remove last HS because cycle incomplete
    if len(hs_L) > 1:
        hs_L = hs_L[:-1]
    if len(hs_R) > 1:
        hs_R = hs_R[:-1]

    # Left step = distance at RIGHT HS; Right step = distance at LEFT HS
    left_steps  = [foot_R[hs, axis] - foot_L[hs, axis] for hs in hs_R]
    right_steps = [foot_L[hs, axis] - foot_R[hs, axis] for hs in hs_L]

    return np.array(left_steps), np.array(right_steps)


# ============================================================
# Plot Functions
def plot_grf(grf_raw, grf_smooth, threshold, title):
    """
    Plot raw/smoothed GRF with threshold line.
    """
    t = np.arange(len(grf_raw))
    plt.figure(figsize=(12,6))
    plt.plot(t, grf_raw, alpha=0.4, label="GRF raw")
    plt.plot(t, grf_smooth, linewidth=2, label="GRF smooth")
    plt.axhline(threshold, color="red", linestyle="--", label="threshold")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_contact_masks(contact1, contact2, title1="Contact 1", title2="Contact 2"):
    """
    Plot two binary contact masks.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(contact1.astype(int))
    axes[0].set_title(title1)
    axes[0].grid(True)

    axes[1].plot(contact2.astype(int))
    axes[1].set_title(title2)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_events(grf_smooth, hs, to, title):
    """
    Plot GRF with HS/TO event markers.
    """
    t = np.arange(len(grf_smooth))
    plt.figure(figsize=(12,6))
    plt.plot(t, grf_smooth, label="GRF smooth")
    plt.scatter(hs, grf_smooth[hs], color="green", label="Heel-strike")
    plt.scatter(to, grf_smooth[to], color="orange", label="Toe-off")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_step_length(left_steps, right_steps):
    """
    Histogram of left/right step lengths.
    """
    plt.figure(figsize=(12,5))
    plt.hist(left_steps, bins=20, alpha=0.6, label="Left steps")
    plt.hist(right_steps, bins=20, alpha=0.6, label="Right steps")
    plt.title("Step length distribution")
    plt.xlabel("Step length [m]")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# Symmetry functions
def symmetry_index(left_value, right_value):
    """
    Compute symmetry index from two scalar values.
    """
    denominator = (abs(left_value) + abs(right_value)) / 2.0
    if denominator < 1e-12:
        return 0.0
    return (100.0 * abs(abs(left_value) - abs(right_value)) / denominator)


def compute_curve_symmetry(left_curve, right_curve):
    """
    Curve-level symmetry metrics (RMSE + normalized diff).
    """
    direct_rmse    = rmse(left_curve, right_curve)
    magnitude_rmse = rmse(np.abs(left_curve), np.abs(right_curve))

    left_norm = np.linalg.norm(left_curve)
    right_norm = np.linalg.norm(right_curve)
    denominator = (left_norm + right_norm) / 2.0

    if denominator < 1e-12:
        normalized_difference = 0.0
    else:
        normalized_difference = np.linalg.norm(left_curve - right_curve) / denominator

    return {
        "direct_RMSE": direct_rmse,
        "magnitude_RMSE": magnitude_rmse,
        "normalized_difference": normalized_difference}


def analyze_temporal_symmetry(df, cycle_left, cycle_right):
    """
    Temporal symmetry: cycle durations + stride frequency.
    """
    has_time = "sim_time" in df.columns

    if has_time:
        t = df["sim_time"].values
        left_durations  = t[cycle_left[1:]] - t[cycle_left[:-1]]
        right_durations = t[cycle_right[1:]] - t[cycle_right[:-1]]
        unit = "s"
    else:
        left_durations  = np.diff(cycle_left)
        right_durations = np.diff(cycle_right)
        unit = "steps"

    left_mean  = float(np.mean(left_durations))
    right_mean = float(np.mean(right_durations))
    left_std  = float(np.std(left_durations))
    right_std = float(np.std(right_durations))
    si = symmetry_index(left_mean, right_mean)

    result = {
        f"Left cycle duration [{unit}]": left_mean,
        f"Left cycle duration SD [{unit}]": left_std,
        f"Right cycle duration [{unit}]": right_mean,
        f"Right cycle duration SD [{unit}]": right_std,
        "Cycle duration SI [%]": si}

    if has_time:
        result["Left stride frequency [strides/min]"] = 60.0 / left_mean
        result["Right stride frequency [strides/min]"] = 60.0 / right_mean

    return result


def compute_stride_length_pelvis(df, cycle_starts, axis=0):
    """
    Compute stride length from pelvis displacement.
    """
    pelvis = df[["pelvis_x", "pelvis_y", "pelvis_z"]].to_numpy(float)
    strides = []
    for i in range(len(cycle_starts) - 1):
        a = cycle_starts[i]
        b = cycle_starts[i + 1]
        stride = pelvis[b, axis] - pelvis[a, axis]
        strides.append(stride)

    strides = np.asarray(strides)
    return float(np.mean(strides)), float(np.std(strides)), strides


def _phase_shift_for(col, side):
    """
    Lookup phase shift for joint/moment curves.
    """
    key_prefix = "knee" if "knee" in col else "hip"
    key = f"{key_prefix}_{side}"
    return PHASE_SHIFT_PERCENT.get(col, PHASE_SHIFT.get(key, 0))


# ============================================================
# MAIN SYMMETRY FUNCTION
def run_symmetry_analysis(df, cycle_left, cycle_right, body_mass_kg=None):
    """
    Full left/right symmetry analysis: temporal, curves, ROM, stride length, and GRF-based metrics.
    """
    results = []

    print("\n" + "=" * 80)
    print("LEFT / RIGHT SYMMETRY ANALYSIS")

    # temporal symmetry 
    temporal = analyze_temporal_symmetry(df, cycle_left, cycle_right)
    for metric, value in temporal.items():
        results.append({"Metric": metric, "Value": value})


    # curve symmetry (ankle, knee, hip) SYMMETRY 
    signal_specs = [
        ("Ankle angle", "ankle_left_angle_deg", "ankle_right_angle_deg", False),
        ("Knee angle", "knee_left_angle_deg", "knee_right_angle_deg", False),
        ("Hip angle", "hip_left_angle_deg", "hip_right_angle_deg", False),
        ("Ankle moment", "ankle_left_moment", "ankle_right_moment", True),
        ("Knee moment", "knee_left_moment", "knee_right_moment", True),
        ("Hip moment", "hip_left_moment", "hip_right_moment", True),
        ("Ankle power", "ankle_left_power", "ankle_right_power", True),
        ("Knee power", "knee_left_power", "knee_right_power", True),
        ("Hip power", "hip_left_power", "hip_right_power", True)]

    for (label, col_left, col_right, mass_norm) in signal_specs:
        sig_left  = df[col_left].values
        sig_right = df[col_right].values

        if mass_norm:
            sig_left = per_kg(sig_left)
            sig_right = per_kg(sig_right)

        mean_left, _  = compute_mean_std(sig_left, cycle_left, flip_sign=should_flip(col_left))
        mean_right, _ = compute_mean_std(sig_right, cycle_right, flip_sign=should_flip(col_right))

        shift_left  = _phase_shift_for(col_left, "l")
        shift_right = _phase_shift_for(col_right, "r")

        mean_left  = apply_phase_shift(mean_left, shift_left)
        mean_right = apply_phase_shift(mean_right, shift_right)

        symmetry = compute_curve_symmetry(mean_left, mean_right)

        results.append({"Metric": f"{label} curve RMSE", "Value": symmetry["direct_RMSE"]})
        results.append({"Metric": f"{label} magnitude RMSE", "Value": symmetry["magnitude_RMSE"]})
        results.append({"Metric": f"{label} normalized difference [%]", "Value": 100 * symmetry["normalized_difference"]})


    # ROM symmetry
    rom_specs = [
        ("Ankle ROM", "ankle_left_angle_deg", "ankle_right_angle_deg"),
        ("Knee ROM", "knee_left_angle_deg", "knee_right_angle_deg"),
        ("Hip ROM", "hip_left_angle_deg", "hip_right_angle_deg")]

    for (label, col_left, col_right) in rom_specs:
        left_mean, left_std, _ = compute_rom_stats(df[col_left].values, cycle_left, flip_sign=should_flip(col_left))
        right_mean, right_std, _ = compute_rom_stats(df[col_right].values, cycle_right, flip_sign=should_flip(col_right))
        si = symmetry_index(left_mean, right_mean)

        results.append({"Metric": f"{label} symmetry index [%]", "Value": si})


    # Stride length symmetry
    left_stride_mean, left_stride_std, _   = compute_stride_length_pelvis(df, cycle_left)
    right_stride_mean, right_stride_std, _ = compute_stride_length_pelvis(df, cycle_right)
    stride_si = symmetry_index(left_stride_mean, right_stride_mean)

    results.append({"Metric": "Stride length symmetry index [%]", "Value": stride_si})


    # GRF-based stance/swing phase and step langth 
    if "grf_left" in df.columns and "grf_right" in df.columns:
        
        # Extract GRF
        grf_L_raw = df["grf_left"].values
        grf_R_raw = df["grf_right"].values

        # Smooth GRF
        grf_L_smooth = smooth_grf(grf_L_raw)
        grf_R_smooth = smooth_grf(grf_R_raw)

        if body_mass_kg is not None:
            threshold = compute_grf_threshold(body_mass_kg)
        else:
            threshold = GRF_THRESHOLD

        # Contact masks
        # contact_L = detect_contact(grf_L_smooth, GRF_THRESHOLD)
        # contact_R = detect_contact(grf_R_smooth, GRF_THRESHOLD)
        contact_L = detect_contact_hysteresis(grf_L_smooth, threshold, threshold * 0.5)
        contact_R = detect_contact_hysteresis(grf_R_smooth, threshold, threshold * 0.5)

        # Events
        hs_L, to_L = detect_events(contact_L)
        hs_R, to_R = detect_events(contact_R)

        # Kinematic events (Zeni et al.)
        foot_L_pos = df[["foot_left_x","foot_left_y","foot_left_z"]].values
        foot_R_pos = df[["foot_right_x","foot_right_y","foot_right_z"]].values
        pelvis_pos = df[["pelvis_x","pelvis_y","pelvis_z"]].values

        hs_L_kin, to_L_kin = detect_events_kinematic(foot_L_pos, pelvis_pos)
        hs_R_kin, to_R_kin = detect_events_kinematic(foot_R_pos, pelvis_pos)

        print("GRF vs kinematic HS left:", len(hs_L), len(hs_L_kin))
        print("GRF vs kinematic HS right:", len(hs_R), len(hs_R_kin))

        # Trim cycles
        hs_L, to_L = trim_cycles(hs_L, to_L)
        hs_R, to_R = trim_cycles(hs_R, to_R)
    
        # Stance/Swing %
        stance_L, swing_L = compute_stance_swing_percent(hs_L, to_L)
        stance_R, swing_R = compute_stance_swing_percent(hs_R, to_R)
    
        # Step length
        foot_L = df[["foot_left_x","foot_left_y","foot_left_z"]].values
        foot_R = df[["foot_right_x","foot_right_y","foot_right_z"]].values
        left_steps, right_steps = compute_step_length(foot_L, foot_R, hs_L, hs_R)


        print("\nGRF-based stance:")
        print(f"  Left stance %: {stance_L}")
        print(f"  Right stance %: {stance_R}")
        print(f"  Mean stance Left:  {np.mean(stance_L):.2f}")
        print(f"  Mean stance Right: {np.mean(stance_R):.2f}")
        print(f"  Std stance Left:   {np.std(stance_L):.2f}")
        print(f"  Std stance Right:  {np.std(stance_R):.2f}")

        print("\nGRF-based swing:")
        print(f"  Left swing %: {swing_L}")
        print(f"  Right swing %: {swing_R}")
        print(f"  Mean swing Left:  {np.mean(swing_L):.2f}")
        print(f"  Mean swing Right: {np.mean(swing_R):.2f}")
        print(f"  Std swing Left:   {np.std(swing_L):.2f}")
        print(f"  Std swing Right:  {np.std(swing_R):.2f}")

        # Step length
        foot_L = df[["foot_left_x","foot_left_y","foot_left_z"]].values
        foot_R = df[["foot_right_x","foot_right_y","foot_right_z"]].values

        # left_steps  = np.array([foot_L[h,0] - foot_R[h,0] for h in hs_L])
        # right_steps = np.array([foot_R[h,0] - foot_L[h,0] for h in hs_R])
        pelvis_x = df["pelvis_x"].values
        foot_L_rel = foot_L[:,0] - pelvis_x
        foot_R_rel = foot_R[:,0] - pelvis_x

        left_steps = np.array([foot_R_rel[h] - foot_L_rel[h] for h in hs_R])
        right_steps = np.array([foot_L_rel[h] - foot_R_rel[h] for h in hs_L])

        print("\nGRF-based step length:")
        print(f"  Left step lengths:  {left_steps}")
        print(f"  Right step lengths: {right_steps}")
        print(f"  Mean Left step:     {np.mean(left_steps):.3f}")
        print(f"  Mean Right step:    {np.mean(right_steps):.3f}")
        print(f"  Std Left step:      {np.std(left_steps):.3f}")
        print(f"  Std Right step:     {np.std(right_steps):.3f}")

        # Symmetry indices
        stance_si = symmetry_index(np.mean(stance_L), np.mean(stance_R))
        swing_si  = symmetry_index(np.mean(swing_L), np.mean(swing_R))
        step_si   = symmetry_index(np.mean(left_steps), np.mean(right_steps))

        print("\nGRF-based symmetry indices:")
        print(f"  Stance symmetry index:      {stance_si:.2f} %")
        print(f"  Swing symmetry index:       {swing_si:.2f} %")
        print(f"  Step length symmetry index: {step_si:.2f} %")

        # Add to results
        results.append({"Metric": "Stance symmetry index [%]", "Value": stance_si})
        results.append({"Metric": "Swing symmetry index [%]", "Value": swing_si})
        results.append({"Metric": "Step length symmetry index [%]", "Value": step_si})

        # ============================================================
        # Debug Plots (shown only in plots-mode)
        args = parse_args()
        if args.plots:
            plot_grf(grf_L_raw, grf_L_smooth, GRF_THRESHOLD, "GRF Left")
            plot_grf(grf_R_raw, grf_R_smooth, GRF_THRESHOLD, "GRF Right")

            plot_contact_masks(contact_L, contact_R, "Contact mask left", "Contact mask right")

            plot_events(grf_L_smooth, hs_L, to_L, "Heel-strike / Toe-off Left")
            plot_events(grf_R_smooth, hs_R, to_R, "Heel-strike / Toe-off Right")

            plot_step_length(left_steps, right_steps)
           
    return pd.DataFrame(results)
