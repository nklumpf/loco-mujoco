import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path


# ── Read measured Data ──────────────────────────────────────────────
script_dir = Path(__file__).parent.resolve()
df_heel = pd.read_csv(script_dir / 'stiffness-data' / 'Heel_Stiffness.csv', sep=';', header=None)
df_heel.columns = ['Displacement', 'Load']
df_keel = pd.read_csv(script_dir / 'stiffness-data' / 'Keel_Stiffness.csv', sep=';', header=None)
df_keel.columns = ['Displacement', 'Load']

# Heel Stiffness Data from Tacca et al. (2023), only size 26 and 28 available
df_heel_26 = pd.read_csv(script_dir / 'stiffness-data' / 'Heel_Size26.csv', sep=';', header=None)
df_heel_26.columns = ['Displacement', 'Load']
df_heel_26['Displacement'] = df_heel_26['Displacement'] * 1000  # m → mm
df_heel_26['Load'] = df_heel_26['Load'] * 1000                  # kN → N
df_heel_28 = pd.read_csv(script_dir / 'stiffness-data' / 'Heel_Size28.csv', sep=';', header=None)
df_heel_28.columns = ['Displacement', 'Load']
df_heel_28['Displacement'] = df_heel_28['Displacement'] * 1000
df_heel_28['Load'] = df_heel_28['Load'] * 1000
# Heel Stiffness Data from Tacca et al. (2023) without shoe, only size 26 and 28 available
df_heel_26_ws = pd.read_csv(script_dir / 'stiffness-data' / 'Heel_Size26_WithoutShoe.csv', sep=';', header=None)
df_heel_26_ws.columns = ['Displacement', 'Load']
df_heel_26_ws['Displacement'] = df_heel_26_ws['Displacement'] * 1000  # m → mm
df_heel_26_ws['Load'] = df_heel_26_ws['Load'] * 1000                  # kN → N
df_heel_28_ws = pd.read_csv(script_dir / 'stiffness-data' / 'Heel_Size28_WithoutShoe.csv', sep=';', header=None)
df_heel_28_ws.columns = ['Displacement', 'Load']
df_heel_28_ws['Displacement'] = df_heel_28_ws['Displacement'] * 1000
df_heel_28_ws['Load'] = df_heel_28_ws['Load'] * 1000

# Keel Stiffness Data from Tacca et al. (2023), only size 26 and 28 available
df_keel_26 = pd.read_csv(script_dir / 'stiffness-data' / 'Keel_Size26.csv', sep=';', header=None)
df_keel_26.columns = ['Displacement', 'Load']
df_keel_26['Displacement'] = df_keel_26['Displacement'] * 1000
df_keel_26['Load'] = df_keel_26['Load'] * 1000
df_keel_28 = pd.read_csv(script_dir / 'stiffness-data' / 'Keel_Size28.csv', sep=';', header=None)
df_keel_28.columns = ['Displacement', 'Load']
df_keel_28['Displacement'] = df_keel_28['Displacement'] * 1000
df_keel_28['Load'] = df_keel_28['Load'] * 1000
# Keel Stiffness Data from Tacca et al. (2023) without shoe, only size 26 and 28 available
df_keel_26_ws = pd.read_csv(script_dir / 'stiffness-data' / 'Keel_Size26_WithoutShoe.csv', sep=';', header=None)
df_keel_26_ws.columns = ['Displacement', 'Load']
df_keel_26_ws['Displacement'] = df_keel_26_ws['Displacement'] * 1000  # m → mm
df_keel_26_ws['Load'] = df_keel_26_ws['Load'] * 1000                  # kN → N
df_keel_28_ws = pd.read_csv(script_dir / 'stiffness-data' / 'Keel_Size28_WithoutShoe.csv', sep=';', header=None)
df_keel_28_ws.columns = ['Displacement', 'Load']
df_keel_28_ws['Displacement'] = df_keel_28_ws['Displacement'] * 1000
df_keel_28_ws['Load'] = df_keel_28_ws['Load'] * 1000

# ── Linear Regression ───────────────────────────────────────────────
def lin_reg(disp, load):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(disp.reshape(-1, 1), load)
    return reg.coef_[0]

# Measured data from Variflex XC (Lecomte)
k_heel     = lin_reg(df_heel['Displacement'].values, df_heel['Load'].values)
k_keel     = lin_reg(df_keel['Displacement'].values, df_keel['Load'].values)

# combine stiffness of heel and keel
disp_comb  = np.concatenate([df_heel['Displacement'].values, df_keel['Displacement'].values])
load_comb  = np.concatenate([df_heel['Load'].values,         df_keel['Load'].values])
k_combined = lin_reg(disp_comb, load_comb)

# Measured data Variflex (Tacca et al. 2023), only size 26 and 28 available, interpolate size 27
k_heel_26  = lin_reg(df_heel_26['Displacement'].values, df_heel_26['Load'].values)
k_heel_28  = lin_reg(df_heel_28['Displacement'].values, df_heel_28['Load'].values)
k_heel_27  = (k_heel_26 + k_heel_28) / 2
# Measured data Variflex without shoe(Tacca et al. 2023), only size 26 and 28 available, interpolate size 27
k_heel_26_ws = lin_reg(df_heel_26_ws['Displacement'].values, df_heel_26_ws['Load'].values)
k_heel_28_ws = lin_reg(df_heel_28_ws['Displacement'].values, df_heel_28_ws['Load'].values)
k_heel_27_ws = (k_heel_26_ws + k_heel_28_ws) / 2

# Measured data Variflex (Tacca et al. 2023), only size 26 and 28 available, interpolate size 27
k_keel_26  = lin_reg(df_keel_26['Displacement'].values, df_keel_26['Load'].values)
k_keel_28  = lin_reg(df_keel_28['Displacement'].values, df_keel_28['Load'].values)
k_keel_27  = (k_keel_26 + k_keel_28) / 2
k_combined_27 = (k_heel_27 + k_keel_27) / 2
# Measured data Variflex without shoe(Tacca et al. 2023), only size 26 and 28 available, interpolate size 27
k_keel_26_ws = lin_reg(df_keel_26_ws['Displacement'].values, df_keel_26_ws['Load'].values)
k_keel_28_ws = lin_reg(df_keel_28_ws['Displacement'].values, df_keel_28_ws['Load'].values)
k_keel_27_ws = (k_keel_26_ws + k_keel_28_ws) / 2
k_combined_27_ws = (k_heel_27_ws + k_keel_27_ws) / 2


x_plot = np.linspace(0.001, max(disp_comb), 300)

# ── Turner Literature Values (linear) ───────────────────────────────
# Stiffness Category 5
k_turner_heel = 37.5
k_turner_keel = 29.1
k_turner_combined = (k_turner_heel + k_turner_keel) / 2

# ── Adamczyk Literature Values (linear) ───────────────────────────────
# subject 90 kg mass
# nominal
k_adamczyk_heel = 53.1
k_adamczyk_keel = 47.5
k_adamczyk_combined = (k_adamczyk_heel + k_adamczyk_keel) / 2


# ── Print Stiffness Values ───────────────────────────────
print(f"\n======================= Linear Heel Stiffness =======================")
print(f"Variflex XC:   k_heel       = {k_heel:.4f} N/mm  (Lecomte)")
print(f"Variflex:      k_heel       = {k_turner_heel:.3f} N/mm  (Turner)")
print(f"Variflex: 90kg k_heel       = {k_adamczyk_heel:.3f} N/mm  (Adamczyk)")
print(f"LP Variflex:   k_heel       = {k_heel_27:.3f} N/mm  (Tacca interpolated)")
print(f"LP Variflex:   k_heel       = {k_heel_27_ws:.3f} N/mm  (Tacca interpolated, without shoe)")

print(f"\n======================= Linear Keel Stiffness =======================")
print(f"Variflex XC:   k_keel       = {k_keel:.4f} N/mm  (Lecomte)")
print(f"Variflex:      k_keel       = {k_turner_keel:.3f} N/mm  (Turner)")
print(f"Variflex: 90kg k_keel       = {k_adamczyk_keel:.3f} N/mm  (Adamczyk)")
print(f"LP Variflex:   k_keel       = {k_keel_27:.3f} N/mm  (Tacca interpolated)")
print(f"LP Variflex:   k_keel       = {k_keel_27_ws:.3f} N/mm  (Tacca interpolated, without shoe)")

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for ax, df_main, df_26, df_28, df_26_ws, df_28_ws, k_main, k_26, k_28, k_27, k_26_ws, k_28_ws, k_27_ws, k_turner, k_adamczyk, color, title in [
    (axes[0], df_heel, df_heel_26, df_heel_28, df_heel_26_ws, df_heel_28_ws,
     k_heel, k_heel_26, k_heel_28, k_heel_27, k_heel_26_ws, k_heel_28_ws, k_heel_27_ws, k_turner_heel, k_adamczyk_heel,
     'blue', 'Heel Stiffness'),
    (axes[1], df_keel, df_keel_26, df_keel_28, df_keel_26_ws, df_keel_28_ws,
     k_keel, k_keel_26, k_keel_28, k_keel_27, k_keel_26_ws, k_keel_28_ws, k_keel_27_ws, k_turner_keel, k_adamczyk_keel,
     'red', 'Keel Stiffness'),
]:
    # Measured data VariFlex XC (Lecomte)
    ax.scatter(df_main['Displacement'], df_main['Load'],
               color='red', s=15, zorder=3, label='VariFlex XC                                                            (Lecomte)')

    # LinReg VariFlex XC (Lecomte)
    ax.plot(x_plot, k_main * x_plot,
            color='red', linestyle='--', linewidth=2,
            label=f'VariFlex XC: LinReg k={k_main:.1f} N/mm                        (Lecomte)')

    # Tacca nonlinear plots Size 26 + 28
    ax.scatter(df_26['Displacement'], df_26['Load'],
               color='green', s=10, alpha=0.5, zorder=3, label='LP VariFlex: Size 26                                                   (Tacca)')
    ax.scatter(df_28['Displacement'], df_28['Load'],
               color='blue', s=10, alpha=0.5, zorder=3, label='LP VariFlex: Size 28                                                   (Tacca)')
    ax.plot(x_plot, k_26 * x_plot,
            color='green', linestyle='-.', linewidth=1.5,
            label=f'LP VariFlex: LinReg Size 26: k={k_26:.1f} N/mm               (Tacca)')
    ax.plot(x_plot, k_28 * x_plot,
            color='blue', linestyle='-.', linewidth=1.5,
            label=f'LP VariFlex: LinReg Size 28: k={k_28:.1f} N/mm               (Tacca)')

    # Tacca Size 27 interpolated
    ax.plot(x_plot, k_27 * x_plot,
            color='violet', linestyle='dotted', linewidth=2,
            label=f'LP VariFlex: LinReg Size 27 (interp.): k={k_27:.1f} N/mm (Tacca)')
    
        # Tacca nonlinear plots Size 26 + 28 WITHOUT SHOE
    ax.scatter(df_26_ws['Displacement'], df_26_ws['Load'],
               color='darkgreen', s=10, alpha=0.5, zorder=3, label='LP VariFlex: Size 26 WS                                             (Tacca)')
    ax.scatter(df_28_ws['Displacement'], df_28_ws['Load'],
               color='darkblue', s=10, alpha=0.5, zorder=3, label='LP VariFlex: Size 28 WS                                             (Tacca)')
    ax.plot(x_plot, k_26_ws * x_plot,
            color='darkgreen', linestyle='-.', linewidth=1.5,
            label=f'LP VariFlex: LinReg Size 26 WS: k={k_26_ws:.1f} N/mm         (Tacca)')
    ax.plot(x_plot, k_28_ws * x_plot,
            color='darkblue', linestyle='-.', linewidth=1.5,
            label=f'LP VariFlex: LinReg Size 28 WS: k={k_28_ws:.1f} N/mm         (Tacca)')

    # Tacca Size 27 interpolated WITHOUT SHOE
    ax.plot(x_plot, k_27_ws * x_plot,
            color='darkviolet', linestyle='dotted', linewidth=2,
            label=f'LP VariFlex: LinReg Size 27 WS: k={k_27_ws:.1f} N/mm         (Tacca)')

    # Turner Literature value (linear, from Paper)
    ax.plot(x_plot, k_turner * x_plot,
            color='orange', linestyle='-', linewidth=2,
            label=f'VariFlex: Literature: k={k_turner} N/mm                          (Turner)')

    # Adamczyk Literature value (linear, from Paper)
    ax.plot(x_plot, k_adamczyk * x_plot,
            color='brown', linestyle='-', linewidth=2,
            label=f'VariFlex: nominal 90kg: k={k_adamczyk} N/mm              (Adamczyk)')
    
    ax.set_xlabel('Displacement [mm]')
    ax.set_ylabel('Load [N]')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=7)
fig.suptitle(
    f'Stiffness VariFlex XC and (LP) VariFlex in Comparison',
    fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig(script_dir / "linear_stiffness_fit.png", dpi=150)
print("\nSaved: linear_stiffness_fit.png")
plt.show()


##################################################################
# --- Deviation Analysis: Lecomte vs. Turner Model ---
# Turner Model: F_turner = k_turner * displacement
def calculate_deviation(df, k_turner):
    """
    Calculates the deviation of the measured data (Lecomte) 
    from the reference model provided by Turner.
    """
    y_pred = k_turner * df['Displacement']
    y_true = df['Load']
    
    # Root Mean Square Error (RMSE) to quantify absolute deviation
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Mean Absolute Percentage Error (MAPE) to quantify relative deviation
    mean_abs_perc_err = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return rmse, mean_abs_perc_err

# Calculate deviations for Heel and Keel components
rmse_h, err_h = calculate_deviation(df_heel, k_turner_heel)
rmse_k, err_k = calculate_deviation(df_keel, k_turner_keel)

print(f"\n======================= Deviation from Turner Model (Lecomte data vs. Turner reference) =======================")
print(f"Heel: RMSE = {rmse_h:.2f} N, Mean Absolute Percentage Error = {err_h:.2f}%")
print(f"Keel: RMSE = {rmse_k:.2f} N, Mean Absolute Percentage Error = {err_k:.2f}%")