"""
Main evaluation script for ESR/SACH prosthesis simulations.

This module orchestrates the full gait-analysis pipeline and depends on:
    - analysis_core.py  (cycle detection, normalization, ROM, phase shifts,
                         literature loading, mean/std computation)
    - symmetry.py       (GRF smoothing, contact detection, HS/TO events,
                         stance/swing %, step length, symmetry metrics)

main_eval loads raw joint/GRF data, trims transients, detects gait cycles,
computes normalized mean/std curves, compares signals against digitized
literature (Pearson r, CCC, RMSE), and generates ROM, GRF, and symmetry
metrics. All results are exported as per-run CSV files plus a central
summary for cross-run comparison.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_core import (
    parse_args,
    get_plot_dir,
    find_latest_run_dir,
    load_literature,
    get_literature_knee_peak,
    detect_knee_cycles,
    trim_cycles,
    joint_stats,
    apply_phase_shift,
    compute_mean_std,
    plot_into_axis,
    compute_rom_stats,
    should_flip,
    N_POINTS,
    HIP_EXTRA_SHIFT,
    RUN_DIAGNOSTICS,
    run_sign_shift_diagnostics,
    PHASE_SHIFT,
    )

from symmetry import (
    run_symmetry_analysis,
    smooth_grf, 
    detect_contact,
    detect_events,
    GRF_THRESHOLD)

from symmetry import trim_cycles as trim_grf_events

# =============================================================================
# single, deduplicated set of literature-comparison helpers (previously
# duplicated 2-3x across active + commented-out blocks in this file).
def concordance_correlation_coefficient(sim, lit):
    """
    Lin's CCC (1989) -- unlike Pearson r, this also penalizes offset/scale
    differences between the two curves, not just shape similarity.
    """
    if np.std(sim) < 1e-9 or np.std(lit) < 1e-9:
        return np.nan
    r = np.corrcoef(sim, lit)[0, 1]
    mean_sim, mean_lit = np.mean(sim), np.mean(lit)
    var_sim, var_lit = np.var(sim), np.var(lit)
    denom = var_sim + var_lit + (mean_sim - mean_lit) ** 2
    if denom < 1e-12:
        return np.nan
    return float((2 * r * np.sqrt(var_sim) * np.sqrt(var_lit)) / denom)


def compare_signal_to_literature(sim_mean, gait_x, lit_y, x_grid):
    """
    Compares a simulated mean curve (already sampled on x_grid, 0-100% GC)
    against a literature curve (gait_x, lit_y) that may only cover a SUBSET
    of the gait cycle -- e.g. GRF literature is often stance-phase only
    (0-60% GC), not the full cycle.

    Restricting the comparison to the x_grid points actually covered by the
    literature's own domain avoids a silent bug: np.interp does not
    extrapolate outside its input range, it CLAMPS to the boundary value.
    Comparing a full 0-100% simulation curve against a stance-only
    literature curve interpolated naively would silently compare the
    swing-phase simulation against a flat line held at the last stance
    value, corrupting every metric below. This affects ANY digitized curve
    that doesn't span the full cycle, not just GRF.

    Returns a dict of metrics plus how much of the gait cycle was actually
    covered, for transparency in the exported CSV.
    """
    domain_min, domain_max = float(np.min(gait_x)), float(np.max(gait_x))
    mask = (x_grid >= domain_min) & (x_grid <= domain_max)

    sim = np.asarray(sim_mean)[mask]
    lit = np.interp(x_grid[mask], gait_x, lit_y)

    if np.std(sim) > 1e-9 and np.std(lit) > 1e-9:
        pearson_r = float(np.corrcoef(sim, lit)[0, 1])
    else:
        pearson_r = np.nan

    return {
        "Pearson_r": pearson_r,
        "CCC": concordance_correlation_coefficient(sim, lit),
        "RMSE_vs_literature": float(np.sqrt(np.mean((sim - lit) ** 2)))}


def signal_unit(label):
    if "angle" in label.lower():
        return "angle_deg"
    elif "moment" in label.lower():
        return "moment_Nm_kg"
    elif "power" in label.lower():
        return "power_W_kg"
    elif "grf" in label.lower():
        return "grf"
    return "other"


def extract_stance_normalized_cycles(grf_raw, n_points=101):
    """
    Schneidet aus dem rohen GRF-Signal jeden Stance-Abschnitt (Heel-Strike
    bis nächster Toe-Off) heraus und resampled ihn auf ein gemeinsames
    0-100%-Stance-Gitter -- dieselbe Konvention, in der die Literaturkurve
    bereits vorliegt. Kein Reskalieren der Literatur nötig.
    """
    grf_smooth = smooth_grf(grf_raw)
    contact = detect_contact(grf_smooth, GRF_THRESHOLD)
    hs, to = detect_events(contact)
    hs, to = trim_grf_events(hs, to)

    curves = []
    for h in hs:
        to_candidates = to[to > h]
        if len(to_candidates) == 0:
            continue
        t = to_candidates[0]
        if t - h < 5:
            continue
        segment = grf_raw[h:t]
        x_orig = np.linspace(0, 100, num=len(segment))
        x_new = np.linspace(0, 100, num=n_points)
        curves.append(np.interp(x_new, x_orig, segment))

    return np.asarray(curves) if curves else np.empty((0, n_points))


def rescale_and_pad_stance_only_literature(gait_x, lit_y, stance_end_pct, drop_width_pct=1.0):
    """
    gait_x läuft 0-100 als % der STANDPHASE (nicht % des Gangzyklus) --
    typisch für digitalisierte reine GRF-Abbildungen, da die
    Schwungphasen-Kraft trivial ~0 ist und oft gar nicht mit abgebildet wird.

    stance_end_pct: der Anteil der Standphase am GESAMTEN Gangzyklus, in %
    (z.B. 62.0 für "Stance = 62% des GC"). Dieser Wert kommt DIREKT aus der
    Literaturquelle (steht praktisch immer als Text im Paper), NICHT aus
    deiner eigenen Simulation -- du positionierst eine bereits vorhandene,
    digitalisierte Kurve auf dem gemeinsamen 0-100%-GC-Raster, unter
    Verwendung DIESER Studie eigenem Stance/Swing-Split, nicht deinem.
    """
    gait_x = np.asarray(gait_x, dtype=float)
    lit_y = np.asarray(lit_y, dtype=float)

    # 0-100 (relativ zur Standphase) -> 0-stance_end_pct (relativ zum GC)
    gait_x_rescaled = gait_x * (stance_end_pct / 100.0)

    drop_end = min(stance_end_pct + drop_width_pct, 100.0)
    x_pad = np.array([drop_end, 100.0])
    y_pad = np.array([0.0, 0.0])
    return np.concatenate([gait_x_rescaled, x_pad]), np.concatenate([lit_y, y_pad])


def main():
    global PHASE_SHIFT

    args = parse_args()

    # NOTE: the raw_joint_data.csv now uses IDENTICAL column names for both ESR and SACH runs (ankle_right_angle_deg, ankle_right_moment, ...).
    # is_esr is only used here to decide whether a reference-trajectory
    # curve exists for the right ankle (it doesn't for ESR, since the ESR
    # hinge has no DOF in the reference trajectory -- ref_ankle_right_angle_deg
    # is NaN in that case).
    is_esr = (args.subtype == "ESR")

    if args.path is not None:
        raw_data_path = args.path
        plot_dir = os.path.dirname(raw_data_path)
    else:
        base_plot_dir = get_plot_dir(args.subtype, args.model_type if args.subtype == "ESR" else None)
        timestamp = args.timestamp or find_latest_run_dir(base_plot_dir)
        plot_dir = get_plot_dir(
            args.subtype,
            args.model_type if args.subtype == "ESR" else None,
            timestamp)
        raw_data_path = os.path.join(plot_dir, "raw_joint_data.csv")

    # Load raw_data path
    print(f"Loading: {raw_data_path}")
    df = pd.read_csv(raw_data_path)


    TRANSIENT_TRIM_STEPS = 150 
    if len(df) > TRANSIENT_TRIM_STEPS:
        df = df.iloc[TRANSIENT_TRIM_STEPS:].reset_index(drop=True)
        print(f"Trimmed the first {TRANSIENT_TRIM_STEPS} steps (reset transient) before analysis.")

    if is_esr:
        df["ankle_right_angle_deg_raw"] = df["ankle_right_angle_deg"].copy()
        df["ankle_right_angle_deg"] = df["ankle_right_angle_deg_lit_convention"]
        df["ankle_right_moment"] = df["ankle_right_moment_corrected"]
        df["ankle_right_power"]  = df["ankle_right_power_corrected"]

    x_norm = np.linspace(0, 100, N_POINTS)

    # Load literature curves
    esr_gait_ankle_angle, esr_lit_ankle_angle   = load_literature("../../gait-data/ESR_ankle_angle.csv")
    esr_gait_ankle_moment, esr_lit_ankle_moment = load_literature("../../gait-data/ESR_ankle_moment.csv")
    esr_gait_ankle_power, esr_lit_ankle_power   = load_literature("../../gait-data/ESR_ankle_power.csv")
    esr_gait_knee_angle, esr_lit_knee_angle     = load_literature("../../gait-data/ESR_knee_angle.csv")
    esr_gait_knee_moment, esr_lit_knee_moment   = load_literature("../../gait-data/ESR_knee_moment.csv")
    esr_gait_knee_power, esr_lit_knee_power     = load_literature("../../gait-data/ESR_knee_power.csv")
    esr_gait_hip_angle, esr_lit_hip_angle       = load_literature("../../gait-data/ESR_hip_angle.csv")
    esr_gait_hip_moment, esr_lit_hip_moment     = load_literature("../../gait-data/ESR_hip_moment.csv")
    esr_gait_hip_power, esr_lit_hip_power       = load_literature("../../gait-data/ESR_hip_power.csv")
    esr_gait_grf, esr_lit_grf                   = load_literature("../../gait-data/ESR_grf.csv")

    gait_ankle_angle, lit_ankle_angle   = load_literature("../../gait-data/Sound_ankle_angle.csv")
    gait_ankle_moment, lit_ankle_moment = load_literature("../../gait-data/Sound_ankle_moment.csv")
    gait_ankle_power, lit_ankle_power   = load_literature("../../gait-data/Sound_ankle_power.csv")
    gait_knee_angle, lit_knee_angle     = load_literature("../../gait-data/Sound_knee_angle.csv")
    gait_knee_moment, lit_knee_moment   = load_literature("../../gait-data/Sound_knee_moment.csv")
    gait_knee_power, lit_knee_power     = load_literature("../../gait-data/Sound_knee_power.csv")
    gait_hip_angle, lit_hip_angle       = load_literature("../../gait-data/Sound_hip_angle.csv")
    gait_hip_moment, lit_hip_moment     = load_literature("../../gait-data/Sound_hip_moment.csv")
    gait_hip_power, lit_hip_power       = load_literature("../../gait-data/Sound_hip_power.csv")
    gait_grf, lit_grf                   = load_literature("../../gait-data/Sound_grf.csv")

    literature_curves = {
        "esr_ankle_angle": (esr_gait_ankle_angle, esr_lit_ankle_angle),
        "esr_ankle_moment": (esr_gait_ankle_moment, esr_lit_ankle_moment),
        "esr_ankle_power": (esr_gait_ankle_power, esr_lit_ankle_power),
        "esr_knee_angle": (esr_gait_knee_angle, esr_lit_knee_angle),
        "esr_knee_moment": (esr_gait_knee_moment, esr_lit_knee_moment),
        "esr_knee_power": (esr_gait_knee_power, esr_lit_knee_power),
        "esr_hip_angle": (esr_gait_hip_angle, esr_lit_hip_angle),
        "esr_hip_moment": (esr_gait_hip_moment, esr_lit_hip_moment),
        "esr_hip_power": (esr_gait_hip_power, esr_lit_hip_power),
        "esr_grf": (esr_gait_grf, esr_lit_grf),
        "sound_ankle_angle": (gait_ankle_angle, lit_ankle_angle),
        "sound_ankle_moment": (gait_ankle_moment, lit_ankle_moment),
        "sound_ankle_power": (gait_ankle_power, lit_ankle_power),
        "sound_knee_angle": (gait_knee_angle, lit_knee_angle),
        "sound_knee_moment": (gait_knee_moment, lit_knee_moment),
        "sound_knee_power": (gait_knee_power, lit_knee_power),
        "sound_hip_angle": (gait_hip_angle, lit_hip_angle),
        "sound_hip_moment": (gait_hip_moment, lit_hip_moment),
        "sound_hip_power": (gait_hip_power, lit_hip_power),
        "sound_grf": (gait_grf, lit_grf)}

    # Determine phase shifts
    knee_shift_r, _ = get_literature_knee_peak(esr_gait_knee_angle, esr_lit_knee_angle)
    hip_shift_r = knee_shift_r + HIP_EXTRA_SHIFT
    knee_shift_l, _ = get_literature_knee_peak(gait_knee_angle, lit_knee_angle)
    hip_shift_l = knee_shift_l + HIP_EXTRA_SHIFT

    PHASE_SHIFT["knee_r"] = knee_shift_r
    PHASE_SHIFT["hip_r"]  = hip_shift_r
    PHASE_SHIFT["knee_l"] = knee_shift_l
    PHASE_SHIFT["hip_l"]  = hip_shift_l

    print(f"Knee phase shift left: {knee_shift_l:.2f}%")
    print(f"Hip phase shift leeft:  {hip_shift_l:.2f}%")
    print(f"Knee phase shift right: {knee_shift_r:.2f}%")
    print(f"Hip phase shift right:  {hip_shift_r:.2f}%")

    # Detect gait cycles
    raw_left    = detect_knee_cycles(df["knee_left_angle_deg"].values)
    raw_right   = detect_knee_cycles(df["knee_right_angle_deg"].values)
    cycle_left  = trim_cycles(raw_left, label="knee_left")
    cycle_right = trim_cycles(raw_right, label="knee_right")

    print(f"\nDetected usable left cycles:  {len(cycle_left) - 1}")
    print(f"Detected usable right cycles: {len(cycle_right) - 1}")

    # Diagnostics
    if RUN_DIAGNOSTICS:
        run_sign_shift_diagnostics(
            df,
            cycle_left,
            cycle_right,
            x_norm,
            literature_curves)

    # ============================================================
    # Joint stats (mean/std curves)
    ref_ankle_angle_left_mean, ref_ankle_angle_left_std, ref_ankle_angle_right_mean, ref_ankle_angle_right_std = joint_stats(
        df, cycle_left, cycle_right, "ref_ankle_left_angle_deg", "ref_ankle_right_angle_deg")

    if is_esr:
        _, _, esr_moment_mean, esr_moment_std = joint_stats(
            df, cycle_left, cycle_right, None, "esr_hinge_moment")
        _, _, esr_power_mean, esr_power_std = joint_stats(
            df, cycle_left, cycle_right, None, "esr_hinge_power")
        _, _, esr_tau_ESR_mean, esr_tau_ESR_std = joint_stats(
        df, cycle_left, cycle_right, None, "esr_tau_ESR")
        _, _, esr_tau_applied_mean, esr_tau_applied_std = joint_stats(
            df, cycle_left, cycle_right, None, "esr_tau_applied")
        _, _, esr_angle_raw_mean, esr_angle_raw_std = joint_stats(
        df, cycle_left, cycle_right, None, "ankle_right_angle_deg_raw")

    ankle_angle_left_mean, ankle_angle_left_std, ankle_angle_right_mean, ankle_angle_right_std = joint_stats(
        df, cycle_left, cycle_right, "ankle_left_angle_deg", "ankle_right_angle_deg")
    ankle_moment_left_mean, ankle_moment_left_std, ankle_moment_right_mean, ankle_moment_right_std = joint_stats(
        df, cycle_left, cycle_right, "ankle_left_moment", "ankle_right_moment", mass_norm=True)
    ankle_power_left_mean, ankle_power_left_std, ankle_power_right_mean, ankle_power_right_std = joint_stats(
        df, cycle_left, cycle_right, "ankle_left_power", "ankle_right_power", mass_norm=True)

    ref_knee_angle_left_mean, ref_knee_angle_left_std, ref_knee_angle_right_mean, ref_knee_angle_right_std = joint_stats(
        df, cycle_left, cycle_right, "ref_knee_left_angle_deg", "ref_knee_right_angle_deg")
    knee_angle_left_mean, knee_angle_left_std, knee_angle_right_mean, knee_angle_right_std = joint_stats(
        df, cycle_left, cycle_right, "knee_left_angle_deg", "knee_right_angle_deg")
    knee_moment_left_mean, knee_moment_left_std, knee_moment_right_mean, knee_moment_right_std = joint_stats(
        df, cycle_left, cycle_right, "knee_left_moment", "knee_right_moment", mass_norm=True)
    knee_power_left_mean, knee_power_left_std, knee_power_right_mean, knee_power_right_std = joint_stats(
        df, cycle_left, cycle_right, "knee_left_power", "knee_right_power", mass_norm=True)

    ref_hip_angle_left_mean, ref_hip_angle_left_std, ref_hip_angle_right_mean, ref_hip_angle_right_std = joint_stats(
        df, cycle_left, cycle_right, "ref_hip_left_angle_deg", "ref_hip_right_angle_deg")
    hip_angle_left_mean, hip_angle_left_std, hip_angle_right_mean, hip_angle_right_std = joint_stats(
        df, cycle_left, cycle_right, "hip_left_angle_deg", "hip_right_angle_deg")
    hip_moment_left_mean, hip_moment_left_std, hip_moment_right_mean, hip_moment_right_std = joint_stats(
        df, cycle_left, cycle_right, "hip_left_moment", "hip_right_moment", mass_norm=True)
    hip_power_left_mean, hip_power_left_std, hip_power_right_mean, hip_power_right_std = joint_stats(
        df, cycle_left, cycle_right, "hip_left_power", "hip_right_power", mass_norm=True)

    # NOTE: suffixed "_bw" (body-weight-normalized) and kept separate from
    # the raw-Newton grf_left_mean/grf_right_mean computed further below via
    # compute_mean_std() for the plots -- both used the same variable name
    # in the previous version of this script, which meant the plotting
    # section silently overwrote the mass-normalized values used here for
    # the literature comparison. Renaming removes that fragile ordering
    # dependency entirely instead of relying on the comparison loop running
    # before the plotting section.
    grf_left_mean_bw, grf_left_std_bw, grf_right_mean_bw, grf_right_std_bw = joint_stats(
        df, cycle_left, cycle_right, "grf_left", "grf_right", mass_norm=True)

    # ============================================================
    # rmse report (raw signals, left vs right) -- kept as-is: this is the
    # RAW signal RMS (not vs. literature), separate from the literature
    # comparison metrics below.
    rmse_results = []
    signal_pairs = [
        ("Ankle angle L", "ankle_left_angle_deg"),
        ("Ankle angle R", "ankle_right_angle_deg"),
        ("Ankle moment L", "ankle_left_moment"),
        ("Ankle moment R", "ankle_right_moment"),
        ("Ankle power L", "ankle_left_power"),
        ("Ankle power R", "ankle_right_power"),
        ("Knee angle L", "knee_left_angle_deg"),
        ("Knee angle R", "knee_right_angle_deg"),
        ("Knee moment L", "knee_left_moment"),
        ("Knee moment R", "knee_right_moment"),
        ("Knee power L", "knee_left_power"),
        ("Knee power R", "knee_right_power"),
        ("Hip angle L", "hip_left_angle_deg"),
        ("Hip angle R", "hip_right_angle_deg"),
        ("Hip moment L", "hip_left_moment"),
        ("Hip moment R", "hip_right_moment"),
        ("Hip power L", "hip_left_power"),
        ("Hip power R", "hip_right_power")]

    for label, col in signal_pairs:
        sig = df[col].values
        rmse_val = np.sqrt(np.mean(sig**2))
        rmse_results.append({"Signal": label, "RMSE": float(rmse_val)})

    rmse_df = pd.DataFrame(rmse_results)
    rmse_csv_path = os.path.join(plot_dir, "rmse_result.csv")
    rmse_df.to_csv(rmse_csv_path, index=False)
    print(f"Saved raw RMSE metrics to: {rmse_csv_path}")

    # ============================================================
    # Literature comparison metrics: Pearson r, CCC, RMSE -- one
    # value per signal, domain-aware (handles literature curves that only
    # cover part of the gait cycle, e.g. stance-only GRF).
    comparison_pairs = [
        ("Ankle angle L", ankle_angle_left_mean,  "sound_ankle_angle"),
        ("Ankle angle R", ankle_angle_right_mean, "esr_ankle_angle"),
        ("Ankle moment L", ankle_moment_left_mean,  "sound_ankle_moment"),
        ("Ankle moment R", ankle_moment_right_mean, "esr_ankle_moment"),
        ("Ankle power L", ankle_power_left_mean,  "sound_ankle_power"),
        ("Ankle power R", ankle_power_right_mean, "esr_ankle_power"),
        ("Knee angle L", knee_angle_left_mean,  "sound_knee_angle"),
        ("Knee angle R", knee_angle_right_mean, "esr_knee_angle"),
        ("Knee moment L", knee_moment_left_mean,  "sound_knee_moment"),
        ("Knee moment R", knee_moment_right_mean, "esr_knee_moment"),
        ("Knee power L", knee_power_left_mean,  "sound_knee_power"),
        ("Knee power R", knee_power_right_mean, "esr_knee_power"),
        ("Hip angle L", hip_angle_left_mean,  "sound_hip_angle"),
        ("Hip angle R", hip_angle_right_mean, "esr_hip_angle"),
        ("Hip moment L", hip_moment_left_mean,  "sound_hip_moment"),
        ("Hip moment R", hip_moment_right_mean, "esr_hip_moment"),
        ("Hip power L", hip_power_left_mean,  "sound_hip_power"),
        ("Hip power R", hip_power_right_mean, "esr_hip_power"),
    ]

    comparison_results = []
    for label, sim_mean, lit_key in comparison_pairs:
        gait_x, lit_y = literature_curves[lit_key]
        metrics = compare_signal_to_literature(sim_mean, gait_x, lit_y, x_norm)
        comparison_results.append({"Signal": label, **metrics})

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df["Unit"] = [signal_unit(label) for label in comparison_df["Signal"]]
    comparison_csv_path = os.path.join(plot_dir, "literature_comparison_metrics.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved literature comparison metrics (Pearson r + CCC + RMSE) to: {comparison_csv_path}")
    print(comparison_df.to_string(index=False))

    grf_stance_x = np.linspace(0, 100, num=101)
    grf_left_stance_curves = extract_stance_normalized_cycles(df["grf_left"].values)
    grf_right_stance_curves = extract_stance_normalized_cycles(df["grf_right"].values)

    for label, stance_curves, lit_key in [
        ("GRF L (stance-normalized)", grf_left_stance_curves, "sound_grf"),
        ("GRF R (stance-normalized)", grf_right_stance_curves, "esr_grf"),
    ]:
        if stance_curves.shape[0] == 0:
            print(f"WARNING: no valid stance cycles found for {label}, skipping.")
            continue
        sim_mean = stance_curves.mean(axis=0)
        metrics = compare_signal_to_literature(sim_mean, gait_x, lit_y, grf_stance_x)
        comparison_results.append({"Signal": label, **metrics})
    # ============================================================
    # Cross-run summary -- ONE row per run, appended to a central CSV (not per-run-folder), so many training runs/checkpoints can be compared
    # side by side. RMSE is kept separate per unit (angle/moment/power/grf)
    # rather than averaged together across incompatible units.
    right_mask = comparison_df["Signal"].str.endswith("R")

    summary_row = {
        "run_date": os.path.basename(os.path.dirname(plot_dir)),
        "run_timestamp": args.timestamp or os.path.basename(plot_dir),
        "subtype": args.subtype,
        "model_type": args.model_type if args.subtype == "ESR" else "",
        "mean_pearson_r": comparison_df["Pearson_r"].mean(skipna=True),
        "mean_ccc": comparison_df["CCC"].mean(skipna=True),
        "mean_pearson_r_right": comparison_df.loc[right_mask, "Pearson_r"].mean(skipna=True),
        "mean_ccc_right": comparison_df.loc[right_mask, "CCC"].mean(skipna=True),
    }
    for unit in ["angle_deg", "moment_Nm_kg", "power_W_kg"]:
        subset = comparison_df[comparison_df["Unit"] == unit]
        summary_row[f"mean_rmse_{unit}"] = subset["RMSE_vs_literature"].mean(skipna=True)
        subset_r = comparison_df[right_mask & (comparison_df["Unit"] == unit)]
        summary_row[f"mean_rmse_{unit}_right"] = subset_r["RMSE_vs_literature"].mean(skipna=True)

    # Zentraler, fester Ort für alle Runs (ESR wie SACH) -- explizit statt
    # über rsplit/".." hergeleitet, damit der Pfad robust bleibt.
    central_summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "run_comparison_summary.csv")

    if os.path.exists(central_summary_path):
        summary_df = pd.read_csv(central_summary_path)
        # denselben Run nicht doppelt eintragen, falls eval_analysis.py
        # mehrfach auf denselben Checkpoint läuft
        summary_df = summary_df[~((summary_df["run_timestamp"] == summary_row["run_timestamp"]) &
                                    (summary_df["run_date"] == summary_row["run_date"]))]
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(central_summary_path, index=False)
    print(f"\nAppended this run's summary to: {central_summary_path}")
    print(summary_df.to_string(index=False))

    # ========================================================
    # === PLOTS: ROM + Ankle/Knee/Hip + GRF ===
    # ROM ANALYSIS
    print("\n" + "=" * 70)
    print("RANGE OF MOTION (ROM)")
    print("=" * 70)

    (ankle_rom_left_mean, ankle_rom_left_std, ankle_rom_left_all)    = compute_rom_stats(df["ankle_left_angle_deg"].values, cycle_left, flip_sign=should_flip("ankle_left_angle_deg"))
    (ankle_rom_right_mean, ankle_rom_right_std, ankle_rom_right_all) = compute_rom_stats(df["ankle_right_angle_deg"].values, cycle_right, flip_sign=should_flip("ankle_right_angle_deg"))

    (knee_rom_left_mean, knee_rom_left_std, knee_rom_left_all)    = compute_rom_stats(df["knee_left_angle_deg"].values, cycle_left, flip_sign=should_flip("knee_left_angle_deg"))
    (knee_rom_right_mean, knee_rom_right_std, knee_rom_right_all) = compute_rom_stats(df["knee_right_angle_deg"].values, cycle_right, flip_sign=should_flip("knee_right_angle_deg"))

    (hip_rom_left_mean, hip_rom_left_std, hip_rom_left_all)    = compute_rom_stats(df["hip_left_angle_deg"].values, cycle_left, flip_sign=should_flip("hip_left_angle_deg"))
    (hip_rom_right_mean, hip_rom_right_std, hip_rom_right_all) = compute_rom_stats(df["hip_right_angle_deg"].values, cycle_right, flip_sign=should_flip("hip_right_angle_deg"))

    lit_ankle_rom_l = (np.max(lit_ankle_angle) - np.min(lit_ankle_angle))
    lit_knee_rom_l  = (np.max(lit_knee_angle) - np.min(lit_knee_angle))
    lit_hip_rom_l   = (np.max(lit_hip_angle) - np.min(lit_hip_angle))
    lit_ankle_rom_r = (np.max(esr_lit_ankle_angle) - np.min(esr_lit_ankle_angle))
    lit_knee_rom_r  = (np.max(esr_lit_knee_angle) - np.min(esr_lit_knee_angle))
    lit_hip_rom_r   = (np.max(esr_lit_hip_angle) - np.min(esr_lit_hip_angle))

    print(f"Ankle ROM Left:  Simulation = {ankle_rom_left_mean:.2f} +/- {ankle_rom_left_std:.2f} deg, Literature = {lit_ankle_rom_l:.2f} deg")
    print(f"Ankle ROM Right: Simulation = {ankle_rom_right_mean:.2f} +/- {ankle_rom_right_std:.2f} deg, Literature = {lit_ankle_rom_r:.2f} deg")
    print(f"Knee ROM Left:  Simulation = {knee_rom_left_mean:.2f} +/- {knee_rom_left_std:.2f} deg, Literature = {lit_knee_rom_l:.2f} deg")
    print(f"Knee ROM Right: Simulation = {knee_rom_right_mean:.2f} +/- {knee_rom_right_std:.2f} deg, Literature = {lit_knee_rom_r:.2f} deg")
    print(f"Hip ROM Left:   Simulation = {hip_rom_left_mean:.2f} +/- {hip_rom_left_std:.2f} deg, Literature = {lit_hip_rom_l:.2f} deg")
    print(f"Hip ROM Right:  Simulation = {hip_rom_right_mean:.2f} +/- {hip_rom_right_std:.2f} deg, Literature = {lit_hip_rom_r:.2f} deg")

    # ========================================================
    # ROM COMPARISON PLOT
    rom_labels = ["Ankle ROM Left", "Ankle ROM Right",
                  "Knee ROM Left", "Knee ROM Right",
                  "Hip ROM Left", "Hip ROM Right"]

    literature_rom = np.array([lit_ankle_rom_l, lit_ankle_rom_r,
                               lit_knee_rom_l, lit_knee_rom_r,
                               lit_hip_rom_l, lit_hip_rom_r])

    simulation_rom = np.array([ankle_rom_left_mean, ankle_rom_right_mean,
                               knee_rom_left_mean, knee_rom_right_mean,
                               hip_rom_left_mean, hip_rom_right_mean])

    simulation_rom_std = np.array([ankle_rom_left_std, ankle_rom_right_std,
                                   knee_rom_left_std, knee_rom_right_std,
                                   hip_rom_left_std, hip_rom_right_std])

    x = np.arange(len(rom_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, literature_rom, width, label="Literature")
    ax.bar(x + width / 2, simulation_rom, width, yerr=simulation_rom_std, capsize=5, label="Simulation")
    ax.set_ylabel("ROM [deg]")
    ax.set_title("Range of Motion: Simulation vs Literature")
    ax.set_xticks(x)
    ax.set_xticklabels(rom_labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    rom_plot_path = os.path.join(plot_dir, "rom_comparison.png")
    plt.savefig(rom_plot_path, dpi=150)
    print(f"Saved ROM comparison plot to: {rom_plot_path}")

    # GRF ANALYSIS
    if ("grf_left" in df.columns and "grf_right" in df.columns):
        (grf_left_mean, grf_left_std)   = compute_mean_std(df["grf_left"].values, cycle_left, flip_sign=False)
        (grf_right_mean, grf_right_std) = compute_mean_std(df["grf_right"].values, cycle_right, flip_sign=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Vertical GRF — Simulation", fontsize=16)

        # NOTE: literature GRF curves (gait_grf/lit_grf, esr_gait_grf/esr_lit_grf)
        # only cover the stance phase, not the full 0-100% gait cycle -- they
        # are plotted here exactly as digitized (no interpolation), so the
        # curve simply ends where the literature data ends. This is expected,
        # not a bug -- GRF is ~0 through swing anyway, and some papers omit
        # that portion entirely rather than plotting a flat zero line.
        axes[0].plot(x_norm, grf_left_mean / 9.81, label="Simulation mean", linewidth=2)
        axes[0].fill_between(x_norm, (grf_left_mean - grf_left_std) / 9.81, (grf_left_mean + grf_left_std) / 9.81, alpha=0.2)
        axes[0].set_title("GRF Left")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(x_norm, grf_right_mean / 9.81, label="Simulation mean", linewidth=2)
        axes[1].fill_between(x_norm, (grf_right_mean - grf_right_std) / 9.81, (grf_right_mean + grf_right_std) / 9.81, alpha=0.2)
        axes[1].set_title("GRF Right")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grf_comparison.png"), dpi=150)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Vertical GRF — Left vs Right (Mean ± SD)", fontsize=16)
        axes[0].plot(x_norm, grf_left_mean, linewidth=2, label="Simulation mean")
        axes[0].fill_between(x_norm, grf_left_mean - grf_left_std, grf_left_mean + grf_left_std, alpha=0.2, label="Simulation ±1 SD")
        axes[0].set_title("Vertical GRF Left")
        axes[0].set_xlabel("Gait Cycle [%]")
        axes[0].set_ylabel("Vertical GRF [N]")
        axes[0].grid(True)
        axes[0].legend(fontsize=8)
        axes[0].set_xlabel("% Stance Phase")
        axes[0].set_ylabel("Vertical Ground Reaction Force (% Body Weight)")

        axes[1].plot(x_norm, grf_right_mean, linewidth=2, label="Simulation mean")
        axes[1].fill_between(x_norm, grf_right_mean - grf_right_std, grf_right_mean + grf_right_std, alpha=0.2, label="Simulation ±1 SD")
        axes[1].set_title("Vertical GRF Right")
        axes[1].set_xlabel("Gait Cycle [%]")
        axes[1].set_ylabel("Vertical GRF [N]")
        axes[1].grid(True)
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel("% Stance Phase")
        axes[1].set_ylabel("Vertical Ground Reaction Force (% Body Weight)")

        plt.tight_layout()
        grf_plot_path = os.path.join(plot_dir, "grf_subplot.png")
        plt.savefig(grf_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Vertical GRF — Simulation (stance-normalized) vs Literature", fontsize=16)

        for ax, stance_curves, gait_x_lit, lit_y_lit, title in [
            (axes[0], grf_left_stance_curves, gait_grf, lit_grf, "GRF Left"),
            (axes[1], grf_right_stance_curves, esr_gait_grf, esr_lit_grf, "GRF Right"),
        ]:
            if stance_curves.shape[0] > 0:
                mean = stance_curves.mean(axis=0) / 9.81
                std = stance_curves.std(axis=0) / 9.81
                ax.plot(grf_stance_x, mean, linewidth=2,
                        label=f"Simulation mean (n={stance_curves.shape[0]} stance phases)")
                ax.fill_between(grf_stance_x, mean - std, mean + std, alpha=0.2)
            ax.plot(gait_x_lit, lit_y_lit, color="black", linewidth=2, label="Literature")
            ax.set_title(title)
            ax.set_xlabel("% Stance Phase")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grf_comparison.png"), dpi=150)
        print(f"\nSaved GRF plot to: {grf_plot_path}")
    else:
        print("\nWARNING: GRF columns not found. Skipping GRF analysis.")

    # Ankle angle/moment/power vs literature
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plot_into_axis(axes[0, 0], x_norm, ankle_angle_left_mean, ankle_angle_left_std, gait_ankle_angle, lit_ankle_angle, title="Ankle angle left", unit="deg")
    plot_into_axis(axes[0, 1], x_norm, ankle_angle_right_mean, ankle_angle_right_std, esr_gait_ankle_angle, esr_lit_ankle_angle, title="Ankle angle right", unit="deg")

    axes[0, 0].plot(x_norm, ref_ankle_angle_left_mean, linestyle=':', color='green', linewidth=2, label='RL goal (ref traj)')
    axes[0, 0].legend(fontsize=8)
    # No reference-trajectory DOF exists for the right ankle under ESR (the ESR hinge is not part of the reference trajectory); plotted here
    # regardless (real data for SACH, the left-side-derived proxy for ESR -- see the joint_stats() call above).
    axes[0, 1].plot(x_norm, ref_ankle_angle_right_mean, linestyle=':', color='green', linewidth=2, label='RL goal (ref traj)')
    if is_esr:
        axes[0, 1].plot(x_norm, -esr_angle_raw_mean, linestyle='-.', color='purple', linewidth=1.5, label='ESR (raw hinge, uncorrected)')
        axes[0, 1].fill_between(x_norm, -(esr_angle_raw_mean - esr_angle_raw_std), -(esr_angle_raw_mean + esr_angle_raw_std), color='purple', alpha=0.15)
    axes[0, 1].legend(fontsize=8)

    plot_into_axis(axes[1, 0], x_norm, ankle_moment_left_mean, ankle_moment_left_std, gait_ankle_moment, lit_ankle_moment, title="Ankle moment left", unit="Nm/kg")
    plot_into_axis(axes[1, 1], x_norm, ankle_moment_right_mean, ankle_moment_right_std, esr_gait_ankle_moment, esr_lit_ankle_moment, title="Ankle moment right", unit="Nm/kg")
    if is_esr:
        axes[1, 1].plot(x_norm, esr_moment_mean, linestyle=':', color='green', linewidth=2, label='ESR')
        axes[1, 1].plot(x_norm, esr_tau_ESR_mean/86.6, linestyle='-.', color='purple', linewidth=1.5, label='tau_ESR (Lecomte law)')
        axes[1, 1].plot(x_norm, esr_tau_applied_mean/86.6, linestyle='-.', color='brown', linewidth=1.5, label='tau_applied (qfrc correction)')
        axes[1, 1].legend(fontsize=8)

    plot_into_axis(axes[2, 0], x_norm, ankle_power_left_mean, ankle_power_left_std, gait_ankle_power, lit_ankle_power, title="Ankle power left", unit="W/kg")
    plot_into_axis(axes[2, 1], x_norm, ankle_power_right_mean, ankle_power_right_std, esr_gait_ankle_power, esr_lit_ankle_power, title="Ankle power right", unit="W/kg")
    if is_esr:
        axes[2, 1].plot(x_norm, esr_power_mean, linestyle=':', color='green', linewidth=2, label='ESR')
        axes[2, 1].legend(fontsize=8)

    plt.tight_layout()
    ankle_plot_path = os.path.join(plot_dir, "ankle_comparison.png")
    plt.savefig(ankle_plot_path)
    print(f"Saved ankle comparison plot to: {ankle_plot_path}")
    plt.close(fig)

    # Knee angle/moment/power vs literature
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plot_into_axis(axes[0, 0], x_norm, knee_angle_left_mean, knee_angle_left_std, gait_knee_angle, lit_knee_angle, title="Knee angle left", unit="deg")
    plot_into_axis(axes[0, 1], x_norm, knee_angle_right_mean, knee_angle_right_std, esr_gait_knee_angle, esr_lit_knee_angle, title="Knee angle right", unit="deg")

    axes[0, 0].plot(x_norm, ref_knee_angle_left_mean, linestyle=':', color='green', linewidth=2, label='RL goal (ref traj)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(x_norm, ref_knee_angle_right_mean, linestyle=':', color='green', linewidth=2, label='RL goal (ref traj)')
    axes[0, 1].legend(fontsize=8)

    plot_into_axis(axes[1, 0], x_norm, knee_moment_left_mean, knee_moment_left_std, gait_knee_moment, lit_knee_moment, title="Knee moment left", unit="Nm/kg")
    plot_into_axis(axes[1, 1], x_norm, knee_moment_right_mean, knee_moment_right_std, esr_gait_knee_moment, esr_lit_knee_moment, title="Knee moment right", unit="Nm/kg")
    plot_into_axis(axes[2, 0], x_norm, knee_power_left_mean, knee_power_left_std, gait_knee_power, lit_knee_power, title="Knee power left", unit="W/kg")
    plot_into_axis(axes[2, 1], x_norm, knee_power_right_mean, knee_power_right_std, esr_gait_knee_power, esr_lit_knee_power, title="Knee power right", unit="W/kg")

    plt.tight_layout()
    knee_plot_path = os.path.join(plot_dir, "knee_comparison.png")
    plt.savefig(knee_plot_path)
    print(f"Saved knee comparison plot to: {knee_plot_path}")
    plt.close(fig)

    # Hip angle/moment/power vs literature
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plot_into_axis(axes[0, 0], x_norm, hip_angle_left_mean, hip_angle_left_std, gait_hip_angle, lit_hip_angle, title="Hip angle left", unit="deg")
    plot_into_axis(axes[0, 1], x_norm, hip_angle_right_mean, hip_angle_right_std, esr_gait_hip_angle, esr_lit_hip_angle, title="Hip angle right", unit="deg")

    axes[0, 0].plot(x_norm, -ref_hip_angle_left_mean, linestyle=':', color='green', linewidth=2, label='RL goal (ref traj)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(x_norm, -ref_hip_angle_right_mean, linestyle=':', color='green', linewidth=2, label='RL goal (ref traj)')
    axes[0, 1].legend(fontsize=8)

    plot_into_axis(axes[1, 0], x_norm, hip_moment_left_mean, hip_moment_left_std, gait_hip_moment, lit_hip_moment, title="Hip moment left", unit="Nm/kg")
    plot_into_axis(axes[1, 1], x_norm, hip_moment_right_mean, hip_moment_right_std, esr_gait_hip_moment, esr_lit_hip_moment, title="Hip moment right", unit="Nm/kg")
    plot_into_axis(axes[2, 0], x_norm, hip_power_left_mean, hip_power_left_std, gait_hip_power, lit_hip_power, title="Hip power left", unit="W/kg")
    plot_into_axis(axes[2, 1], x_norm, hip_power_right_mean, hip_power_right_std, esr_gait_hip_power, esr_lit_hip_power, title="Hip power right", unit="W/kg")

    plt.tight_layout()
    hip_plot_path = os.path.join(plot_dir, "hip_comparison.png")
    plt.savefig(hip_plot_path)
    print(f"Saved hip comparison plot to: {hip_plot_path}")
    plt.close(fig)

    # === Symmetry analysis ===
    symmetry_df = run_symmetry_analysis(df, cycle_left, cycle_right)
    symmetry_csv_path = os.path.join(plot_dir, "symmetry_result.csv")
    symmetry_df.to_csv(symmetry_csv_path, index=False)
    print(f"Saved symmetry metrics to: {symmetry_csv_path}")

    si_rows = symmetry_df[symmetry_df["Metric"].str.contains("symmetry index")]
    if not si_rows.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(si_rows["Metric"], si_rows["Value"])
        ax.set_ylabel("Symmetry index [%]")
        ax.set_xticks(np.arange(len(si_rows)))
        ax.set_xticklabels(si_rows["Metric"], rotation=45, ha="right")
        ax.grid(True, axis="y")
        plt.tight_layout()
        sym_plot_path = os.path.join(plot_dir, "symmetry_comparison.png")
        plt.savefig(sym_plot_path)
        print(f"Saved symmetry comparison plot to: {sym_plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()