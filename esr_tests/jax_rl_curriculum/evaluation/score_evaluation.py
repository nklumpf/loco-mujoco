"""
Run Comparison and Ranking Script
=================================

This script analyzes the evaluation results of multiple reinforcement learning runs stored in a CSV file.

It performs the following steps:
1. Loads the evaluation summary from a CSV file.
2. Combines the run date and timestamp into a single datetime value.
3. Identifies the best run for each individual evaluation metric.
4. Normalizes all metrics to a common scale between 0 and 1, accounting for whether higher or lower values indicate better performance.
5. Calculates an overall score as the unweighted mean of the normalized metrics for each run.
6. Ranks all runs according to their overall scores.
7. Prints the top 10 runs and the individual metrics of the best overall run.

Metrics related to correlation are maximized, whereas metrics describing the root mean square error (RMSE) are minimized.

The overall score provides a relative comparison between the evaluated runs. It depends on the set of runs included in the CSV file and does not represent an absolute measure of gait quality.
"""

import pandas as pd

# Load evaluation data
df = pd.read_csv("run_comparison_summary.csv")

# Combine the run date and timestamp into a single datetime column.
df["run_datetime"] = pd.to_datetime(df["run_date"] + " " + df["run_timestamp"],
                                    format="%Y-%m-%d %H-%M-%S")


# Define evaluation metrics

# Metrics to maximize: higher values indicate better agreement between simulated and reference gait data.
maximize_cols = [
    "mean_pearson_r",
    "mean_ccc",
    "mean_pearson_r_right",
    "mean_ccc_right"]

# Metrics to minimize: lower RMSE values indicate smaller deviations between simulated and reference gait data.
minimize_cols = [
    "mean_rmse_angle_deg",
    "mean_rmse_angle_deg_right",
    "mean_rmse_moment_Nm_kg",
    "mean_rmse_moment_Nm_kg_right",
    "mean_rmse_power_W_kg",
    "mean_rmse_power_W_kg_right",]

# Identify the best run for each metric
print("\n" + "=" * 80)
print("BEST RUN PER METRIC")
print("=" * 80)

# Find the run with the highest value for each correlation metric.
for col in maximize_cols:
    idx = df[col].idxmax()
    row = df.loc[idx]

    print(f"\n{col}")
    print(f"  Value     : {row[col]:.6f}")
    print(f"  Run time  : {row['run_datetime']}")

# Find the run with the lowest value for each RMSE metric.
for col in minimize_cols:
    idx = df[col].idxmin()
    row = df.loc[idx]

    print(f"\n{col}")
    print(f"  Value     : {row[col]:.6f}")
    print(f"  Run time  : {row['run_datetime']}")


# Calculate the overall score
score_df = pd.DataFrame(index=df.index)

# Normalize metrics to the range [0, 1], where 1 represents the best value among all evaluated runs.
for col in maximize_cols:
    min_val = df[col].min()
    max_val = df[col].max()

    if max_val == min_val:
        # Assign the same maximum score if all runs have identical values.
        score_df[col] = 1.0
    else:
        score_df[col] = ((df[col] - min_val) / (max_val - min_val))

# For metrics to minimize, lower values receive higher scores.
for col in minimize_cols:
    min_val = df[col].min()
    max_val = df[col].max()

    if max_val == min_val:
        # Assign the same maximum score if all runs have identical values.
        score_df[col] = 1.0
    else:
        score_df[col] = ((max_val - df[col]) / (max_val - min_val))

# Calculate the unweighted mean of all normalized metrics.
# Each metric contributes equally to the overall score.
df["overall_score"] = score_df.mean(axis=1)

# =====================
# Rank all runs
# Sort the runs in descending order of their overall scores.
ranking = (
    df.sort_values("overall_score", ascending=False)
      .reset_index(drop=True))

print("\n" + "=" * 80)
print("TOP 10 RUNS BY OVERALL SCORE")
print("=" * 80)

# Display the ten highest-ranked runs.
for i, row in ranking.head(10).iterrows():
    print(
        f"{i+1:2d}. "
        f"{row['run_datetime']}  "
        f"Score = {row['overall_score']:.4f}")

# Display the best overall run
best = ranking.iloc[0]

print("\n" + "=" * 80)
print("BEST OVERALL RUN")
print("=" * 80)

print(f"Run time  : {best['run_datetime']}")
print(f"Score     : {best['overall_score']:.4f}")

# Print all individual evaluation metrics for the best-ranked run.
print("\nIndividual metric values:")
for col in maximize_cols + minimize_cols:
    print(f"{col:30s}: {best[col]:.6f}")