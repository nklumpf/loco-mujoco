#!/usr/bin/env python3
"""
Erzeugt alle CSV-Dateien für die Alignment-Sensitivitäts-Abbildung.

Eingabe:  eine CSV mit den Sweep-Läufen PLUS dem nominalen Lauf (ohne Offset).
          Die Datei darf mit oder ohne Kopfzeile vorliegen. Die Offsets werden
          aus der Spalte run_timestamp gelesen, z. B. "07-14-55_pos0.005_0_ori0_0_0".
          Die Zeile ohne "_pos" im Namen ist der nominale Lauf.
Ausgabe:  tables/align_{right,left}_{pos1,pos2,ori1,ori3}.csv
          Spalten: offset, r, ccc, angle, moment, power
          offset: Position in mm, Orientierung in Grad (Nominal = 0)
          Metriken: relative Änderung gegenüber Nominal in %, positiv = besser
          (r, CCC: Anstieg = besser; RMSE: Abnahme = besser)

Aufruf:   python make_alignment_tables.py
          python make_alignment_tables.py --input data/alignment_sweep.csv --outdir tables
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["run_date", "run_timestamp", "subtype", "model_type",
        "mean_pearson_r", "mean_ccc", "mean_pearson_r_right", "mean_ccc_right",
        "mean_rmse_angle_deg", "mean_rmse_angle_deg_right",
        "mean_rmse_moment_Nm_kg", "mean_rmse_moment_Nm_kg_right",
        "mean_rmse_power_W_kg", "mean_rmse_power_W_kg_right",
        "mean_pearson_r_left", "mean_ccc_left",
        "mean_rmse_angle_deg_left", "mean_rmse_moment_Nm_kg_left",
        "mean_rmse_power_W_kg_left"]

# Reihenfolge der Offsets im Namen: pos<a>_<b>_ori<x>_<y>_<z>
PARAMS = ["pos1", "pos2", "ori1", "ori2", "ori3"]
SCALE = [1000, 1000, 180 / np.pi, 180 / np.pi, 180 / np.pi]  # m -> mm, rad -> Grad
PATTERN = re.compile(r"pos(-?[\d.]+)_(-?[\d.]+)_ori(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)")


def load(path):
    first = Path(path).read_text().splitlines()[0]
    if first.startswith("run_date"):
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=COLS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="alignment_sweep.csv")
    ap.add_argument("--outdir", default="tables")
    args = ap.parse_args()

    df = load(args.input)
    is_sweep = df["run_timestamp"].astype(str).str.contains("_pos")
    nominal = df[~is_sweep]
    assert len(nominal) == 1, f"Erwarte genau eine nominale Zeile, gefunden: {len(nominal)}"
    nom = nominal.iloc[0]
    sweep = df[is_sweep]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for side in ("right", "left"):
        # Spalte -> (Name im Plot, größer = besser?)
        metrics = {
            f"mean_pearson_r_{side}":         ("r",      True),
            f"mean_ccc_{side}":               ("ccc",    True),
            f"mean_rmse_angle_deg_{side}":    ("angle",  False),
            f"mean_rmse_moment_Nm_kg_{side}": ("moment", False),
            f"mean_rmse_power_W_kg_{side}":   ("power",  False),
        }
        rows = []
        for _, r in sweep.iterrows():
            vals = [float(x) for x in PATTERN.search(r["run_timestamp"]).groups()]
            nonzero = [i for i, x in enumerate(vals) if x != 0]
            assert len(nonzero) == 1, f"Mehr als ein Offset in {r['run_timestamp']}"
            k = nonzero[0]
            row = {"param": PARAMS[k], "offset": round(vals[k] * SCALE[k], 1)}
            for col, (name, higher_is_better) in metrics.items():
                base = nom[col]
                diff = (r[col] - base) if higher_is_better else (base - r[col])
                row[name] = 100 * diff / base
            rows.append(row)
        out = pd.DataFrame(rows)

        for param, g in out.groupby("param"):
            zero = {"param": param, "offset": 0.0, **{n: 0.0 for n, _ in metrics.values()}}
            g = pd.concat([g, pd.DataFrame([zero])]).sort_values("offset")
            path = outdir / f"align_{side}_{param}.csv"
            g.drop(columns="param").to_csv(path, index=False)
            print("geschrieben:", path)


if __name__ == "__main__":
    main()