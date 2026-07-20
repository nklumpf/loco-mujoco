import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Define the quadratic function: F(x) = a*x^2 + b*x
def quad_func(x, a, b):
    return a * x**2 + b * x

# Load the data
script_dir = Path(__file__).parent.resolve()
heel_file = script_dir / 'stiffness-data' / 'Heel_Stiffness.csv'
keel_file = script_dir / 'stiffness-data' / 'Keel_Stiffness.csv'
df_heel = pd.read_csv(heel_file, sep=';', header=None, names=['Displacement', 'Load'])
df_keel = pd.read_csv(keel_file, sep=';', header=None, names=['Displacement', 'Load'])

# Calculate non-linear fit parameters
popt_h, _ = curve_fit(quad_func, df_heel['Displacement'], df_heel['Load'])
popt_k, _ = curve_fit(quad_func, df_keel['Displacement'], df_keel['Load'])

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot Heel (Blue) ---
# Plot raw data (split by index)
axes[0].plot(df_heel['Displacement'], df_heel['Load'], 
             color='blue', marker='o', linestyle='', markersize=4, alpha=0.3, label='Measurement (Lecomte)')
# Plot non-linear fit
x_h = np.linspace(0, df_heel['Displacement'].max(), 100)
axes[0].plot(x_h, quad_func(x_h, *popt_h), color='darkblue', linewidth=2, 
             label=f'Non-linear Fit: {popt_h[0]:.6f}$x^2$ + {popt_h[1]:.6f}$x$')

axes[0].set_title('Heel Stiffness')
axes[0].set_xlabel('Displacement [mm]')
axes[0].set_ylabel('Load [N]')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# --- Plot Keel (Red) ---
# Plot raw data (split by index)
axes[1].plot(df_keel['Displacement'], df_keel['Load'], 
             color='red', marker='o', linestyle='', markersize=4, alpha=0.3, label='Measurement (Lecomte)')
# Plot non-linear fit
x_k = np.linspace(0, df_keel['Displacement'].max(), 100)
axes[1].plot(x_k, quad_func(x_k, *popt_k), color='darkred', linewidth=2, 
             label=f'Non-linear Fit: {popt_k[0]:.6f}$x^2$ + {popt_k[1]:.6f}$x$')

axes[1].set_title('Keel Stiffness')
axes[1].set_xlabel('Displacement [mm]')
axes[1].set_ylabel('Load [N]')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.suptitle('Stiffness Loading VariFlex XC (Data & Non-linear Fit)')
plt.tight_layout()
plt.savefig(script_dir / "nonlinear_stiffness_fit.png", dpi=150)
print("Saved: nonlinear_stiffness_fit.png")
plt.show()