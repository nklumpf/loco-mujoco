import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define script directory for saving the plot
script_dir = Path(__file__).parent.resolve()

# Define range: 0 to 20 degrees converted to radians
theta_deg = np.linspace(0, 20, 100)
x = np.radians(theta_deg)

# Define exact functions and their small-angle approximations
sin_x = np.sin(x)
approx_sin = x

cos_x = np.cos(x)
approx_cos = 1 - (x**2 / 2)

# Calculation of relative errors (avoid division by zero at 0°)
rel_err_sin = np.abs((approx_sin - sin_x) / sin_x)
rel_err_sin[0] = 0  # Handle limit at x=0
rel_err_cos = np.abs((approx_cos - cos_x) / cos_x)

# Plotting the results
plt.figure(figsize=(12, 5))

# Subplot for Sinus
plt.subplot(1, 2, 1)
plt.plot(theta_deg, sin_x, label="sin(x)")
plt.plot(theta_deg, approx_sin, '--', label="x (Approx)")
plt.title("Small-angle approximation for sin(x)")
plt.xlabel("Angle (degrees)")
plt.legend()
plt.grid(True)

# Subplot for Cosinus
plt.subplot(1, 2, 2)
plt.plot(theta_deg, cos_x, label="cos(x)")
plt.plot(theta_deg, approx_cos, '--', label="1 - x²/2 (Approx)")
plt.title("Small-angle approximation for cos(x)")
plt.xlabel("Angle (degrees)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(script_dir / "small_angle_approximation.png", dpi=150)
print("\nPlot saved: small_angle_approximation.png")
plt.show()

# Print detailed error analysis for 20 degrees
print("-" * 30)
print(f"Error Analysis at 20°:")
print(f"sin(20°): {sin_x[-1]:.6f}")
print(f"Relative error sin(x): {rel_err_sin[-1]:.6%}")
print("-" * 30)
print(f"cos(20°): {cos_x[-1]:.6f}")
print(f"Relative error cos(x): {rel_err_cos[-1]:.6%}")
print("-" * 30)