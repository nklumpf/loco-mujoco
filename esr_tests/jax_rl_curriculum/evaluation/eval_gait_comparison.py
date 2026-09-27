"""
Comparison of gait-cycle kinematics and kinetics for:
    - Literature
    - SACH
    - ESR linear
    - ESR nonlinear

The analysis follows the same methodology as gait_analysis.py:
    1. Load raw joint data
    2. Detect gait cycles using knee-flexion minima
    3. Remove warm-up cycles
    4. Normalize each cycle to 0-100% gait cycle
    5. Apply signal sign conventions
    6. Mass-normalize moments and powers
    7. Apply phase alignment based on literature knee flexion
    8. Compute mean +/- SD
    9. Plot Literature + SACH + ESR linear + ESR nonlinear for knee and hip angle, moment and power
"""

# IMPORTS
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks


# Config
BODY_MASS_KG = 86.6
NORMALIZE_MOMENT_POWER_BY_MASS = True

# Gait-cycle detection
MIN_PEAK_DISTANCE = 80
MIN_PEAK_PROMINENCE = 5

# Remove initial cycles
N_WARMUP_CYCLES = 2

# Remove final incomplete cycle
DROP_LAST_CYCLE = True

# Normalized gait-cycle resolution
N_POINTS = 200

# Same additional hip shift as in gait_analysis.py
HIP_EXTRA_SHIFT = 5


# data directories
DATASETS = {
    "SACH": os.path.join("plots","SACH"),
    "ESR linear": os.path.join("plots","ESR","linear"),
    "ESR nonlinear": os.path.join("plots","ESR","nonlinear"),}

# output directory
OUTPUT_DIR = os.path.join("plots","comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# sign conventions
FLIP_SIGN_OVERRIDE = {
    "hip_angle_left_deg": False,
    "hip_angle_right_deg": False,
    "hip_power_left": False,
    "hip_power_right": False,
    "knee_moment_right": False,
    "knee_power_right": False,}


def should_flip(column_name: str) -> bool:
    """
    Determine whether a signal needs sign inversion.
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
    # Hip kinetics
    if ("hip_moment" in name or "hip_power" in name):
        return True

    # Knee moment/power
    return False


def detect_knee_cycles(knee_angle):
    """
    Detect gait-cycle boundaries using maximum knee flexion.
    MuJoCo convention:
        flexion = negative angle
    Therefore:
        find_peaks(-knee_angle)
    identifies maximum knee flexion.
    """
    peaks, _ = find_peaks(-knee_angle, distance=MIN_PEAK_DISTANCE, prominence=MIN_PEAK_PROMINENCE)

    return peaks


def trim_cycles(raw_starts, label=""):
    """
    Remove warm-up cycles and optionally the final cycle.
    """
    if DROP_LAST_CYCLE:
        trimmed = raw_starts[N_WARMUP_CYCLES:-1]

    else:
        trimmed = raw_starts[N_WARMUP_CYCLES:]

    if len(trimmed) < 2:
        raise ValueError(f"Not enough gait cycles detected "
                         f"for '{label}'.")

    return trimmed


def normalize_cycle(signal, start, end):
    """
    Extract one gait cycle and interpolate it onto a uniform 0-100% grid.
    """
    cyc = signal[start:end]

    x_original = np.linspace(0, 100, len(cyc))
    x_norm = np.linspace(0, 100, N_POINTS)

    return np.interp(x_norm, x_original, cyc)


def compute_mean_std(signal, cycle_starts, flip_sign=False):
    """
    Compute mean and standard deviation across normalized gait cycles.
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

    return (mean_curve, std_curve)


# mass normalization
def per_kg(signal):
    return signal / BODY_MASS_KG


def apply_phase_shift(curve, percent):
    """
    Circular phase shift in percent gait cycle.
    """
    shift = int(round(percent / 100.0* len(curve)))
    return np.roll(curve, shift)


def load_literature(path):
    """
    Load literature CSV:
        column 0 = gait cycle [%]
        column 1 = signal
    """
    df = pd.read_csv(path, sep=",", header=None)
    gait = df.iloc[:, 0].values
    values = df.iloc[:, 1].values
    idx = np.argsort(gait)

    return (gait[idx], values[idx])


def get_literature_knee_peak(gait_lit, knee_angle_lit):
    """
    Determine the gait-cycle position of maximum knee flexion in literature.
    """
    idx = np.argmax(knee_angle_lit)

    return (gait_lit[idx], knee_angle_lit[idx])


def load_all_literature():
    """
    Load the six literature reference curves.
    """
    literature = {}
    literature["knee_angle"]  = load_literature("../gait-data/ESR_knee_angle.csv")
    literature["knee_moment"] = load_literature("../gait-data/ESR_knee_moment.csv")
    literature["knee_power"]  = load_literature("../gait-data/ESR_knee_power.csv")
    literature["hip_angle"]   = load_literature("../gait-data/ESR_hip_angle.csv")
    literature["hip_moment"]  = load_literature("../gait-data/ESR_hip_moment.csv")
    literature["hip_power"]   = load_literature("../gait-data/ESR_hip_power.csv")
    return literature


def analyze_dataset(name, data_dir, knee_shift, hip_shift):
    """
    Analyze one simulation dataset.
    Returns a dictionary containing mean/std curves for all six signals and both legs.
    """
    print("\n" + "=" * 70)
    print(f"Analyzing: {name}")
    print("=" * 70)
    csv_path = os.path.join(data_dir, "raw_joint_data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find:\n{csv_path}")

    df = pd.read_csv(csv_path)

    # Detect gait cycles
    raw_left  = detect_knee_cycles(df["knee_angle_left_deg"].values)
    raw_right = detect_knee_cycles(df["knee_angle_right_deg"].values)

    cycle_left  = trim_cycles(raw_left, label=f"{name} left")
    cycle_right = trim_cycles(raw_right, label=f"{name} right")

    print(f"Detected usable left cycles: "
          f"{len(cycle_left) - 1}")
    print(f"Detected usable right cycles: "
          f"{len(cycle_right) - 1}")


    # Helper
    def get_joint_stats(col_left, col_right, mass_normalize=False, shift=0):
        sig_left  = df[col_left].values
        sig_right = df[col_right].values

        if mass_normalize:
            sig_left  = per_kg(sig_left)
            sig_right = per_kg(sig_right)

        mean_left, std_left   = compute_mean_std(sig_left, cycle_left, flip_sign=should_flip(col_left))
        mean_right, std_right = compute_mean_std(sig_right, cycle_right, flip_sign=should_flip(col_right))

        mean_left  = apply_phase_shift(mean_left, shift)
        mean_right = apply_phase_shift(mean_right, shift)

        std_left  = apply_phase_shift(std_left, shift)
        std_right = apply_phase_shift(std_right, shift)

        return {
            "left_mean": mean_left,
            "left_std": std_left,
            "right_mean": mean_right,
            "right_std": std_right,}

    # --------------------------------------------------------
    # Knee
    knee_angle = get_joint_stats(
        "knee_angle_left_deg",
        "knee_angle_right_deg",
        mass_normalize=False,
        shift=knee_shift)

    knee_moment = get_joint_stats(
        "knee_moment_left",
        "knee_moment_right",
        mass_normalize=True,
        shift=knee_shift)

    knee_power = get_joint_stats(
        "knee_power_left",
        "knee_power_right",
        mass_normalize=True,
        shift=knee_shift)

    # --------------------------------------------------------
    # Hip
    hip_angle = get_joint_stats(
        "hip_angle_left_deg",
        "hip_angle_right_deg",
        mass_normalize=False,
        shift=hip_shift)

    hip_moment = get_joint_stats(
        "hip_moment_left",
        "hip_moment_right",
        mass_normalize=True,
        shift=hip_shift)

    hip_power = get_joint_stats(
        "hip_power_left",
        "hip_power_right",
        mass_normalize=True,
        shift=hip_shift)

    return {
        "knee_angle": knee_angle,
        "knee_moment": knee_moment,
        "knee_power": knee_power,
        "hip_angle": hip_angle,
        "hip_moment": hip_moment,
        "hip_power": hip_power,}


def plot_comparison_axis(ax, x_norm, literature_gait, literature_values, datasets, signal_name, side, title, unit):
    """
    Plot literature + all simulation datasets on one axis.
    """
    # Literature
    ax.plot(literature_gait, literature_values, linestyle="--", linewidth=2.5, label="Literature")

    # Simulation datasets
    for dataset_name, data in datasets.items():
        stats = data[signal_name]
        mean  = stats[f"{side}_mean"]
        std   = stats[f"{side}_std"]

        ax.plot(x_norm, mean, linewidth=2, label=dataset_name)
        ax.fill_between(x_norm, mean - std, mean + std, alpha=0.12)

    ax.set_title(title)
    ax.set_xlabel("Gait Cycle [%]")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)


def create_knee_plot(datasets, literature, x_norm):
    """
    Create 3x2 knee comparison plot.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Knee Joint Comparison", fontsize=17)

    # --------------------------------------------------------
    # Knee angle
    plot_comparison_axis(
        axes[0, 0],
        x_norm,
        literature["knee_angle"][0],
        literature["knee_angle"][1],
        datasets,
        "knee_angle",
        "left",
        "Knee Angle Left",
        "Angle [deg]")

    plot_comparison_axis(
        axes[0, 1],
        x_norm,
        literature["knee_angle"][0],
        literature["knee_angle"][1],
        datasets,
        "knee_angle",
        "right",
        "Knee Angle Right",
        "Angle [deg]")

    # --------------------------------------------------------
    # Knee moment
    plot_comparison_axis(
        axes[1, 0],
        x_norm,
        literature["knee_moment"][0],
        literature["knee_moment"][1],
        datasets,
        "knee_moment",
        "left",
        "Knee Moment Left",
        "Moment [Nm/kg]")

    plot_comparison_axis(
        axes[1, 1],
        x_norm,
        literature["knee_moment"][0],
        literature["knee_moment"][1],
        datasets,
        "knee_moment",
        "right",
        "Knee Moment Right",
        "Moment [Nm/kg]")

    # --------------------------------------------------------
    # Knee power
    plot_comparison_axis(
        axes[2, 0],
        x_norm,
        literature["knee_power"][0],
        literature["knee_power"][1],
        datasets,
        "knee_power",
        "left",
        "Knee Power Left",
        "Power [W/kg]")

    plot_comparison_axis(
        axes[2, 1],
        x_norm,
        literature["knee_power"][0],
        literature["knee_power"][1],
        datasets,
        "knee_power",
        "right",
        "Knee Power Right",
        "Power [W/kg]")

    # --------------------------------------------------------
    # Common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_path = os.path.join(OUTPUT_DIR,"knee_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


def create_hip_plot(datasets, literature, x_norm):
    """
    Create 3x2 hip comparison plot.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Hip Joint Comparison", fontsize=17)

    # --------------------------------------------------------
    # Hip angle
    plot_comparison_axis(
        axes[0, 0],
        x_norm,
        literature["hip_angle"][0],
        literature["hip_angle"][1],
        datasets,
        "hip_angle",
        "left",
        "Hip Angle Left",
        "Angle [deg]")

    plot_comparison_axis(
        axes[0, 1],
        x_norm,
        literature["hip_angle"][0],
        literature["hip_angle"][1],
        datasets,
        "hip_angle",
        "right",
        "Hip Angle Right",
        "Angle [deg]")

    # --------------------------------------------------------
    # Hip moment
    plot_comparison_axis(
        axes[1, 0],
        x_norm,
        literature["hip_moment"][0],
        literature["hip_moment"][1],
        datasets,
        "hip_moment",
        "left",
        "Hip Moment Left",
        "Moment [Nm/kg]")

    plot_comparison_axis(
        axes[1, 1],
        x_norm,
        literature["hip_moment"][0],
        literature["hip_moment"][1],
        datasets,
        "hip_moment",
        "right",
        "Hip Moment Right",
        "Moment [Nm/kg]")

    # --------------------------------------------------------
    # Hip power
    plot_comparison_axis(
        axes[2, 0],
        x_norm,
        literature["hip_power"][0],
        literature["hip_power"][1],
        datasets,
        "hip_power",
        "left",
        "Hip Power Left",
        "Power [W/kg]")

    plot_comparison_axis(
        axes[2, 1],
        x_norm,
        literature["hip_power"][0],
        literature["hip_power"][1],
        datasets,
        "hip_power",
        "right",
        "Hip Power Right",
        "Power [W/kg]")

    # --------------------------------------------------------
    # Common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_path = os.path.join(OUTPUT_DIR, "hip_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
def main():
    print("\n"+ "=" * 80)
    print("GAIT COMPARISON")
    print("=" * 80)

    # Normalized x-axis
    x_norm = np.linspace(0, 100, N_POINTS)

    # Load literature
    literature = load_all_literature()

    # Determine phase shift from literature
    knee_shift, knee_peak = get_literature_knee_peak(literature["knee_angle"][0], literature["knee_angle"][1])

    hip_shift = (knee_shift + HIP_EXTRA_SHIFT)

    print(f"\nLiterature knee-flexion peak: "
          f"{knee_peak:.2f} deg")
    print(f"Knee phase shift: "
          f"{knee_shift:.2f}%")
    print(f"Hip phase shift: "
          f"{hip_shift:.2f}%")

    # --------------------------------------------------------
    # Analyze all simulation datasets
    datasets = {}

    for dataset_name, data_dir in DATASETS.items():
        datasets[dataset_name] = analyze_dataset(dataset_name, data_dir, knee_shift, hip_shift)

    # --------------------------------------------------------
    # Create plots
    create_knee_plot(datasets, literature, x_norm)
    create_hip_plot(datasets, literature, x_norm)

    print("\n"+ "=" * 80)
    print("Comparison finished.")
    print(f"Output directory:\n{OUTPUT_DIR}")
    print("=" * 80)


# ============================================================
# RUN
if __name__ == "__main__":
    main()