"""
Diagnostic comparison of MuJoCo generalized joint forces vs. mimic torque
and gait-analysis torque (from raw_joint_data.csv).

Compares per joint (hip/knee, left/right):
    - mimic torque sensor (from torque_diagnostic.csv)
    - gait-analysis torque (hip/knee_moment_* from raw_joint_data.csv)
    - qfrc_actuator
    - qfrc_constraint
    - qfrc_passive
    - qfrc_applied
    - sum of generalized-force contributions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

PROSTHESIS_SUBTYPE = "ESR"      # "ESR" or "SACH"
ESR_MODEL_TYPE = "linear"       # "linear", "nonlinear", or None for SACH

BODY_MASS_KG = 86.6
MASS_NORMALIZE = False          # True -> Nm/kg, False -> Nm

def get_output_dir(prosthesis_subtype, esr_model_type=None):
    base_dir = "plots"
    out_dir = os.path.join(base_dir, prosthesis_subtype)
    if prosthesis_subtype == "ESR" and esr_model_type is not None:
        out_dir = os.path.join(out_dir, esr_model_type)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

OUTPUT_DIR = get_output_dir(PROSTHESIS_SUBTYPE, ESR_MODEL_TYPE)

TORQUE_DIAG_CSV = os.path.join(OUTPUT_DIR, "torque_diagnostic.csv")
RAW_JOINT_CSV   = os.path.join(OUTPUT_DIR, "raw_joint_data.csv")

# joint mapping: names -> column names in both CSVs
JOINTS = {
    "hip_left": {
        "mimic_col": "hip_moment_left_mimic",
        "raw_col":   "hip_moment_left",
        "qfrc_cols": [
            "hip_left_qfrc_actuator",
            "hip_left_qfrc_constraint",
            "hip_left_qfrc_passive",
            "hip_left_qfrc_applied",
        ],
    },
    "hip_right": {
        "mimic_col": "hip_moment_right_mimic",
        "raw_col":   "hip_moment_right",
        "qfrc_cols": [
            "hip_right_qfrc_actuator",
            "hip_right_qfrc_constraint",
            "hip_right_qfrc_passive",
            "hip_right_qfrc_applied",
        ],
    },
    "knee_left": {
        "mimic_col": "knee_moment_left_mimic",
        "raw_col":   "knee_moment_left",
        "qfrc_cols": [
            "knee_left_qfrc_actuator",
            "knee_left_qfrc_constraint",
            "knee_left_qfrc_passive",
            "knee_left_qfrc_applied",
        ],
    },
    "knee_right": {
        "mimic_col": "knee_moment_right_mimic",
        "raw_col":   "knee_moment_right",
        "qfrc_cols": [
            "knee_right_qfrc_actuator",
            "knee_right_qfrc_constraint",
            "knee_right_qfrc_passive",
            "knee_right_qfrc_applied",
        ],
    },
}

# ============================================================
# Helpers
# ============================================================

def maybe_mass_normalize(arr):
    if MASS_NORMALIZE:
        return arr / BODY_MASS_KG
    return arr

def load_data():
    if not os.path.exists(TORQUE_DIAG_CSV):
        raise FileNotFoundError(f"Missing {TORQUE_DIAG_CSV}")
    if not os.path.exists(RAW_JOINT_CSV):
        raise FileNotFoundError(f"Missing {RAW_JOINT_CSV}")

    df_diag = pd.read_csv(TORQUE_DIAG_CSV)
    df_raw  = pd.read_csv(RAW_JOINT_CSV)

    # ensure both have a 'step' column
    if "step" not in df_diag.columns:
        df_diag["step"] = np.arange(len(df_diag))
    if "step" not in df_raw.columns:
        df_raw["step"] = np.arange(len(df_raw))

    df = pd.merge(df_diag, df_raw, on="step", how="inner")
    return df

def plot_joint(df, joint_name):
    meta = JOINTS[joint_name]
    mimic_col = meta["mimic_col"]
    raw_col   = meta["raw_col"]
    qfrc_cols = meta["qfrc_cols"]

    x = df["step"].values

    plt.figure(figsize=(12, 6))

    # mimic torque (sensor)
    if mimic_col in df.columns:
        y_mimic = maybe_mass_normalize(df[mimic_col].values)
        plt.plot(x, y_mimic, label="mimic torque sensor", linewidth=1.8, color="black")

    # gait-analysis torque (raw_joint_data)
    if raw_col in df.columns:
        y_raw = maybe_mass_normalize(df[raw_col].values)
        plt.plot(x, y_raw, label="gait-analysis torque (raw_joint_data)", linewidth=1.5, color="tab:blue")

    # generalized forces
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    labels = ["qfrc_actuator", "qfrc_constraint", "qfrc_passive", "qfrc_applied"]
    sum_components = np.zeros(len(df))

    for col, lab, c in zip(qfrc_cols, labels, colors):
        if col in df.columns:
            y = maybe_mass_normalize(df[col].values)
            sum_components += y
            plt.plot(x, y, label=lab, linewidth=1.0, alpha=0.7, color=c)

    # sum of generalized forces
    plt.plot(x, sum_components, label="sum(qfrc_*)", linewidth=1.5, linestyle="--", color="gray")

    unit = "Nm/kg" if MASS_NORMALIZE else "Nm"
    plt.xlabel("Simulation step")
    plt.ylabel(f"Joint torque / generalized force [{unit}]")
    plt.title(f"Joint torque vs. generalized forces — {joint_name}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    fname = os.path.join(OUTPUT_DIR, f"{joint_name}_torque_vs_generalized_forces.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.show()

def print_statistics(df):
    print("\n" + "=" * 90)
    print("GENERALIZED FORCE DIAGNOSTICS (mean/std/min/max)")
    print("=" * 90)

    for joint_name, meta in JOINTS.items():
        print(f"\n{joint_name}")
        print("-" * 60)

        cols = meta["qfrc_cols"] + [meta["mimic_col"], meta["raw_col"]]
        for col in cols:
            if col not in df.columns:
                continue
            values = maybe_mass_normalize(df[col].values)
            print(
                f"{col:30s}: "
                f"mean={np.mean(values): .4f}, "
                f"std={np.std(values): .4f}, "
                f"min={np.min(values): .4f}, "
                f"max={np.max(values): .4f}, "
                f"max_abs={np.max(np.abs(values)): .4f}"
            )

# ============================================================
# Main
# ============================================================

def main():
    df = load_data()
    print(f"Loaded {len(df)} merged samples from:")
    print(f"  {TORQUE_DIAG_CSV}")
    print(f"  {RAW_JOINT_CSV}")

    print_statistics(df)

    for joint in JOINTS.keys():
        plot_joint(df, joint)

if __name__ == "__main__":
    main()