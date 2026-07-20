"""
Measures the SACH foot STL bounding box and computes the
ESR pylon correction value (how much shorter the pylon must be when switching from SACH to ESR foot).
"""

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

# Paths and known values 
SACH_STL_PATH = "/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/sach.stl"
SCALE = 0.001           # MuJoCo scale=[0.001, 0.001, 0.001]
ESR_HEIGHT_M = 0.160    # Vari-Flex XC stack height in meters (known)

# Load mesh
print("Loading SACH mesh...")
mesh = trimesh.load(SACH_STL_PATH)

# Raw bounds (in original STL units, likely mm)
bounds = mesh.bounds   # shape (2, 3): [[xmin,ymin,zmin], [xmax,ymax,zmax]]
extents = mesh.bounding_box.extents  # [dx, dy, dz]

print("\n=== RAW STL (before scale) ===")
print(f"  Bounds min  : x={bounds[0,0]:.2f}  y={bounds[0,1]:.2f}  z={bounds[0,2]:.2f}")
print(f"  Bounds max  : x={bounds[1,0]:.2f}  y={bounds[1,1]:.2f}  z={bounds[1,2]:.2f}")
print(f"  Extents     : x={extents[0]:.2f}  y={extents[1]:.2f}  z={extents[2]:.2f}")

# Scaled bounds (meters, as seen in MuJoCo)
bounds_m  = bounds  * SCALE
extents_m = extents * SCALE

print("\n=== SCALED (MuJoCo units, meters) ===")
print(f"  Bounds min  : x={bounds_m[0,0]:.4f}  y={bounds_m[0,1]:.4f}  z={bounds_m[0,2]:.4f}")
print(f"  Bounds max  : x={bounds_m[1,0]:.4f}  y={bounds_m[1,1]:.4f}  z={bounds_m[1,2]:.4f}")
print(f"  Extents (m) : x={extents_m[0]:.4f}  y={extents_m[1]:.4f}  z={extents_m[2]:.4f}")

# Same Rotation as in MuJoCo-Code: quat=[w,x,y,z]=[0.643,0,-0.766,0]
q = [0, -0.766, 0, 0.643] 
R = Rotation.from_quat(q).as_matrix()
# rotate Vertices
rotated_vertices = mesh.vertices @ R.T

# Bounding Box after Rotation
min_v = rotated_vertices.min(axis=0)
max_v = rotated_vertices.max(axis=0)
extents_rotated = (max_v - min_v) * 0.001  # m

# In your model the vertical axis is Y, but the STL might be oriented differently.
print("\n=== HEIGHT CANDIDATES ===")
axis_labels = ["X", "Y", "Z"]
for i, label in enumerate(axis_labels):
    h_raw = extents_rotated[i]*1000
    h_m   = extents_rotated[i]
    print(f"  {label}-axis height: {h_m*1000:.1f} mm  ({h_m:.4f} m)")

# Best guess: smallest axis is the stack height
dominant_axis = int(np.argmin(extents_rotated))
sach_height_m = extents_rotated[dominant_axis]
print(f"\n  → Relevant axis: {axis_labels[dominant_axis]}  =  {sach_height_m*1000:.1f} mm")

# ── Pylon correction 
correction = ESR_HEIGHT_M - sach_height_m

print("\n=== ESR PYLON CORRECTION ===")
print(f"  ESR stack height  : {ESR_HEIGHT_M*1000:.1f} mm")
print(f"  SACH stack height : {sach_height_m*1000:.1f} mm  (axis {axis_labels[dominant_axis]})")
print(f"  Correction needed : {correction*1000:.1f} mm  ({correction:.4f} m)")


########################################################
## VISUALIZATION CORRECTION FACTOR
########################################################
"""
Measures the PYLON STL bounding box to compute visualization scaling factor for ESR correction.
"""


# Paths and known values 
PYLON_STL_PATH = "/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/pylon.stl"
SCALE = 0.001           # MuJoCo scale=[0.001, 0.001, 0.001]

# Load mesh 
print("Loading PYLON mesh...")
mesh = trimesh.load(PYLON_STL_PATH)

# Raw bounds (in original STL units, likely mm) 
bounds = mesh.bounds   # shape (2, 3): [[xmin,ymin,zmin], [xmax,ymax,zmax]]
extents = mesh.bounding_box.extents  # [dx, dy, dz]

print("\n=== RAW STL (before scale) ===")
print(f"  Bounds min  : x={bounds[0,0]:.2f}  y={bounds[0,1]:.2f}  z={bounds[0,2]:.2f}")
print(f"  Bounds max  : x={bounds[1,0]:.2f}  y={bounds[1,1]:.2f}  z={bounds[1,2]:.2f}")
print(f"  Extents     : x={extents[0]:.2f}  y={extents[1]:.2f}  z={extents[2]:.2f}")

# Scaled bounds (meters, as seen in MuJoCo) 
bounds_m  = bounds  * SCALE
extents_m = extents * SCALE

print("\n=== SCALED (MuJoCo units, meters) ===")
print(f"  Bounds min  : x={bounds_m[0,0]:.4f}  y={bounds_m[0,1]:.4f}  z={bounds_m[0,2]:.4f}")
print(f"  Bounds max  : x={bounds_m[1,0]:.4f}  y={bounds_m[1,1]:.4f}  z={bounds_m[1,2]:.4f}")
print(f"  Extents (m) : x={extents_m[0]:.4f}  y={extents_m[1]:.4f}  z={extents_m[2]:.4f}")

# height = largest extent (the foot's tallest dimension) 
print("\n=== HEIGHT CANDIDATES ===")
axis_labels = ["X", "Y", "Z"]
for i, label in enumerate(axis_labels):
    h_raw = extents[i]
    h_m   = extents_m[i]
    print(f"  {label}-axis height: {h_raw:.2f} mm raw  →  {h_m*1000:.1f} mm  ({h_m:.4f} m)")

# Best guess: tallest axis is the stack height
dominant_axis = int(np.argmax(extents))
pylon_height_m = extents_m[dominant_axis]
print(f"\n  → Largest axis: {axis_labels[dominant_axis]}  =  {pylon_height_m*1000:.1f} mm")

# Pylon correction
pylon_diff = pylon_height_m - correction
print(f"  → New Pylon height after correction: {pylon_diff*1000:.1f} mm  ({pylon_diff:.4f} m)")

# New scaling factor in y-direction for visualization
print("\n=== PYLON SCALING FACTOR ===")
scale_factor_y = pylon_diff / pylon_height_m * SCALE
print(f" → Visualization scaling factor for ESR correction: {scale_factor_y:.6f} (multiply y-coordinates by this factor)")