"""
esr_hinge.py

Lecomte-inspired two-segment ESR foot model (forefoot and rearfoot).

Structure:

    RearFoot (free in space, no ground contact)
        |
        Hinge (Functional Joint Center after Lecomte)
        |
    ForeFoot (ground contact)
        |
      Ground

Geometry (based on Lecomte / Vari-Flex XC data):
    FJC x = 110 mm anterior to heel
    FJC y =  55 mm above sole
    RearFoot length = 110 mm
    ForeFoot length = 160 mm
    Total foot length = 270 mm

ESR torque law (small-angle rotational spring):
    k_theta = k_lin * L^2       [Nm/rad]
    tau     = -k_theta * theta

Tests:
    1. Passive restitution:  start at theta=10deg, check return to 0
    2. Impact test:          foot falls, GRF measured
    3. Roll-over test:       forward velocity, observe rolling
    4. Torque-angle:         plot tau vs theta
"""

import matplotlib
matplotlib.use("Agg")

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time as time_module
from pathlib import Path

# ============================================================
# PARAMETERS
# ============================================================
dt       = 0.001          # [s]
sim_time = 2.0           # [s]

k_lin    = 30000.0        # linear stiffness value            [N/m] 
c_lin    = 10             # damping value                     [Ns/m]
L        = 0.11           # heel -> FJC (Vari-Flex XC: 110mm) [m]    
k_theta  = k_lin * L**2   # [Nm/rad]
c_theta  = c_lin * L**2   # [Nsm/rad]

# Initial conditions
theta0_deg     = -15.0     # initial hinge angle [deg]
vx0            = 0         # forward velocity    [m/s]
vz0            = -0.2      # downward velocity   [m/s]

# ============================================================
# XML
# ============================================================

xml = rf"""
<mujoco model="esr_hinge">

    <option gravity="0 0 -9.81"
            timestep="{dt}"/>

    <default>
        <geom
            solref="0.02 1"
            solimp="0.9 0.95 0.0015"
            friction="1.0 0.05 0.01"/>
    </default>

    <worldbody>

        <!-- FLOOR -->
        <geom
            name="floor"
            type="plane"
            size="10 10 0.1"/>

        <!-- REARFOOT: free in space, no ground contact (blue) -->
        <body name="rearfoot" pos="0 0 0.15">

            <!-- Freejoint: full 6-DOF freedom -->
            <freejoint name="rearfoot_free"/>

            <!-- Rearfoot geometry: centered at FJC, extends backwards -->
            <geom
                name="rearfoot_geom"
                type="box"
                size="0.055 0.05 0.025"
                pos="-0.055 0 0"
                mass="0.25"
                rgba="0.3 0.3 0.8 1"
                contype="0"
                conaffinity="0"/>

            <!-- Hinge site at FJC -->
            <site
                name="hinge_site"
                pos="0 0 0"
                size="0.015"
                rgba="1 1 0 1"/>

            <!-- FOREFOOT: rotates around FJC, has ground contact -->
            <body name="forefoot" pos="0 0 0">

                <joint
                    name="esr_hinge"
                    type="hinge"
                    axis="0 1 0"
                    damping="0"
                    limited="true"
                    range="-0.7 0.7"/>

                <!-- forefoot geometry: 160mm long, centered 80mm forward of FJC -->
                <geom
                    name="forefoot_geom"
                    type="box"
                    size="0.08 0.05 0.025"
                    pos="0.08 0 0"
                    mass="0.35"
                    rgba="0.8 0.3 0.2 1"
                    contype="1"
                    conaffinity="1"/>

            </body>

        </body>

    </worldbody>

    <!-- SENSORS -->
    <sensor>

        <!-- ESR joint kinematics -->
        <jointpos 
            name="esr_hinge_pos" 
            joint="esr_hinge"/>
        <jointvel 
            name="esr_hinge_vel" 
            joint="esr_hinge"/>

        <!-- Force and torque at ankle site (between pylon and foot) -->
        <force    
            name="hinge_force"  
            site="hinge_site"/>
        <torque   
            name="hinge_torque" 
            site="hinge_site"/>

    </sensor>

</mujoco>
"""

# ============================================================
# LOAD MODEL
# ============================================================
model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# ============================================================
# IDs
# ============================================================
# Joints
# ESR joint
esr_hinge_joint   = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_JOINT, 
    "esr_hinge"
)
esr_hinge_qpos    = model.jnt_qposadr[esr_hinge_joint]
esr_hinge_dof     = model.jnt_dofadr[esr_hinge_joint]

# Bodies
rearfoot_body = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_BODY, 
    "rearfoot"
)
forefoot_body = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_BODY, 
    "forefoot"
)

# Sensors
def get_sensor_adr(model, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return model.sensor_adr[sensor_id]

adr_pos    = get_sensor_adr(model, "esr_hinge_pos")
adr_vel    = get_sensor_adr(model, "esr_hinge_vel")
adr_force  = get_sensor_adr(model, "hinge_force")
adr_torque = get_sensor_adr(model, "hinge_torque")

print(f"hinge_qpos = {esr_hinge_qpos}")
print(f"hinge_dof  = {esr_hinge_dof}")
print(f"Joint range: {model.jnt_range[esr_hinge_joint]}")


# ============================================================
# INITIAL CONDITIONS
# ============================================================
def reset_initial_conditions(data):
    mujoco.mj_resetData(model, data)
    data.qpos[esr_hinge_qpos]  = np.deg2rad(theta0_deg)
    # freejoint velocities: qvel[0:6] = [vx, vy, vz, wx, wy, wz]
    data.qvel[0] = vx0 # forward
    data.qvel[2] = vz0 # downward
    mujoco.mj_forward(model, data)

reset_initial_conditions(data) 


# ============================================================
# LOGGING
# ============================================================
n_steps = int(sim_time / dt)

# Initialization
time_log         = np.zeros(n_steps)    # time                   [s]
theta_log        = np.zeros(n_steps)    # joint angle            [rad]
theta_dot_log    = np.zeros(n_steps)    # joint velocity         [rad/s]
esr_torque_log   = np.zeros(n_steps)    # restoring torque       [Nm]
esr_force_log    = np.zeros(n_steps)    # applied ESR force      [N]
z_equiv_log      = np.zeros(n_steps)    # ESR joint displacement [m]
grf_log          = np.zeros(n_steps)    # ground reaction force  [N]
contact_log      = np.zeros(n_steps)    # number of contacts
rearfoot_z_log   = np.zeros(n_steps)    # rearfoot z-position    [m]
forefoot_z_log   = np.zeros(n_steps)    # forefoot z-position    [m]
rearfoot_x_log   = np.zeros(n_steps)    # rearfoot x-position    [m]

# Sensor logs
sensor_pos_log    = np.zeros(n_steps)      # esr_slide_pos
sensor_vel_log    = np.zeros(n_steps)      # esr_slide_vel
sensor_force_log  = np.zeros((n_steps,3))  # ankle_force  [Fx, Fy, Fz]
sensor_torque_log = np.zeros((n_steps,3))  # ankle_torque [Tx, Ty, Tz]


# ============================================================
# SIMULATION
# ============================================================
print(f"\nRunning ESR hinge model")
print(f"theta0={theta0_deg}deg, vx={vx0}m/s, vz={vz0}m/s")

for i in range(n_steps):

    # --- ESR force law (Linear Kelvin-Voigt Model with Conversion to Hinge Joint) ---
    theta     = data.qpos[esr_hinge_qpos]
    theta_dot = data.qvel[esr_hinge_dof]
    # ESR rotational spring
    tau   = -k_theta * theta - c_theta * theta_dot
    F_esr = -tau / L
    z_equiv = L * theta
    # Apply torque in joint
    data.qfrc_applied[:] = 0
    data.qfrc_applied[esr_hinge_dof] = tau

    # --- Step ---
    mujoco.mj_step(model, data)

    # --- GRF from contacts ---
    grf_z = 0.0
    for c in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, force)
        grf_z += force[0]

    # --- Log ---
    time_log[i]         = data.time
    theta_log[i]        = theta
    theta_dot_log[i]    = theta_dot
    esr_torque_log[i]   = tau
    # Conversion to slide joint forces, vertical displacement
    esr_force_log[i]    = F_esr
    z_equiv_log[i]      = z_equiv
    grf_log[i]          = grf_z
    contact_log[i]      = data.ncon
    rearfoot_z_log[i]   = data.xpos[rearfoot_body, 2]
    forefoot_z_log[i]   = data.xpos[forefoot_body, 2]
    rearfoot_x_log[i]   = data.xpos[rearfoot_body, 0]

    # --- Sensor readout ---
    sensor_pos_log[i]     = data.sensordata[adr_pos]
    sensor_vel_log[i]     = data.sensordata[adr_vel]
    sensor_force_log[i]   = data.sensordata[adr_force:adr_force+3]
    sensor_torque_log[i]  = data.sensordata[adr_torque:adr_torque+3]

print("Done")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle(
    f"ESR Hinge Joint\n"
    f"k_lin={k_lin:.0f} N/m,  L={L*1000:.0f}mm,  "
    f"k_theta={k_theta:.1f} Nm/rad  |  tau=-k_theta*theta",
    fontsize=11
)

# Body positions
axes[0, 0].plot(time_log, rearfoot_z_log, label="rearfoot")
axes[0, 0].plot(time_log, forefoot_z_log, label="forefoot")
axes[0, 0].set_title("Vertical positions")
axes[0, 0].set_xlabel("Time [s]")
axes[0, 0].set_ylabel("Z position [m]")
axes[0, 0].legend()
axes[0, 0].grid()

# Hinge angle
axes[0, 1].plot(time_log, np.rad2deg(theta_log))
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].set_title("Hinge angle")
axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].set_ylabel("theta [deg]")
axes[0, 1].grid()

# Torque vs angle
axes[1, 0].plot(np.rad2deg(theta_log), esr_torque_log, linewidth=1, alpha=0.8)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
axes[1, 0].set_title("Torque-Angle")
axes[1, 0].set_xlabel("theta [deg]")
axes[1, 0].set_ylabel("tau [Nm]")
axes[1, 0].grid()

# GRF
axes[1, 1].plot(time_log, grf_log)
axes[1, 1].set_title("Ground reaction force")
axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].set_ylabel("GRF [N]")
axes[1, 1].grid()

# Forward position
axes[2, 0].plot(time_log, rearfoot_x_log)
axes[2, 0].set_title("Forward position")
axes[2, 0].set_xlabel("Time [s]")
axes[2, 0].set_ylabel("X position [m]")
axes[2, 0].grid()

# ESR force vs GRF
axes[2, 1].plot(time_log, esr_force_log, label="ESR F_esr")
axes[2, 1].plot(time_log, grf_log,   label="GRF")
axes[2, 1].set_title("ESR force vs GRF")
axes[2, 1].set_xlabel("Time [s]")
axes[2, 1].set_ylabel("Force [N]")
axes[2, 1].legend()
axes[2, 1].grid()

plt.tight_layout()
script_dir = Path(__file__).parent.resolve()
plt.savefig(script_dir / "esr_hinge.png", dpi=150)
print("Saved: esr_hinge.png")


# ============================================================
# INTERACTIVE VIEWER
# ============================================================
print("\nStarting interactive viewer...")

# Reset for start
mujoco.mj_resetData(model, data)
reset_initial_conditions(data)

# with mujoco.viewer.launch_passive(model, data) as viewer:
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    # Camera onto model
    viewer.cam.lookat[:] = [0.2, 0, 0.1]
    viewer.cam.distance  = 1.5
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -15

    while viewer.is_running():
        # Force computation and apply it
        theta     = data.qpos[esr_hinge_qpos]
        theta_dot = data.qvel[esr_hinge_dof]
        tau       = -k_theta * theta - c_theta * theta_dot
        data.qfrc_applied[:] = 0
        data.qfrc_applied[esr_hinge_dof] = tau

        mujoco.mj_step(model, data)

        # Synchronize Viewer
        viewer.sync()

        # Pause to see it slower
        time_module.sleep(0.01)