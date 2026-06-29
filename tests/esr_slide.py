"""
esr_slide.py

Toy ESR model with a slide joint (Rigney et al. 2018)

Structure
      PYLON (free in x and z)
        |
        |  ESR SLIDE JOINT
        |
      FOOT
        |
      FLOOR
      
Linear Kelvin-Voigt equation as joint force: 
    F_z = -k * z
    
Force is applied through qfrc_applied.

Ground reaction force with standard parameters.

Sensors:
  - Joint position and velocity of ESR slide
  - Force and torque at ankle site (between pylon and foot)
  - GRF via mj_contactForce
"""

#=============================================================
# IMPORTS
#=============================================================
import matplotlib
matplotlib.use("Agg")

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

#=============================================================
# PARAMETERS
#=============================================================
sim_time = 10.0     # [s]
dt       = 0.001    # [s]

k_lin    = 10000    # linear stiffness value [N/m]
# TODO: Add damping? More stable in xml-File
c_lin    = 0        # damping value          [Ns/m]

# Initial conditions
z0            = 0   # initial position    [m]


#=============================================================
# XML-FILE
#=============================================================
xml = rf"""
<mujoco model="esr_slide">

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

        <!-- PYLON -->
        <body
            name="pylon"
            pos="0 0 0.8">

            <joint
                name="pylon_z"
                type="slide"
                axis="0 0 1"/>
            <joint 
                name="pylon_x" 
                type="slide" 
                axis="1 0 0"/>

            <geom
                name="pylon_geom"
                type="box"
                size="0.04 0.04 0.20"
                mass="30"
                rgba="0.2 0.4 0.8 1"/>

            <!-- Ankle Site: between pylon and foot, for force/torque sensor -->
            <site
                name="ankle_site"
                pos="0 0 -0.20"
                size="0.01"/>

            <!-- FOOT -->
            <body
                name="foot"
                pos="0 0 -0.25">

                <joint
                    name="esr_slide"
                    type="slide"
                    axis="0 0 1"
                    damping="100"
                    limited="true"
                    range="-0.05 0.05"/>

                <geom
                    name="foot_geom"
                    type="box"
                    size="0.135 0.05 0.03"
                    mass="0.594"
                    rgba="0.8 0.3 0.2 1"/>

            </body>

        </body>

    </worldbody>

    <!-- SENSORS -->
    <sensor>

        <!-- ESR joint kinematics -->
        <jointpos
            name="esr_slide_pos"
            joint="esr_slide"/>
        <jointvel
            name="esr_slide_vel"
            joint="esr_slide"/>

        <!-- Force and torque at ankle site (between pylon and foot) -->
        <force
            name="ankle_force"
            site="ankle_site"/>
        <torque
            name="ankle_torque"
            site="ankle_site"/>

    </sensor>

</mujoco>
"""

# ============================================================
# LOAD MODEL
# ============================================================
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)


# ============================================================
# IDS
# ============================================================
# Joints
# ESR joint
esr_slide_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "esr_slide"
)
esr_slide_qpos = model.jnt_qposadr[esr_slide_joint]
esr_slide_dof  = model.jnt_dofadr[esr_slide_joint]

# Pylon joint
pylon_x_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "pylon_x"
)
pylon_x_dof = model.jnt_dofadr[pylon_x_joint]
pylon_z_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "pylon_z"
)
pylon_z_dof = model.jnt_dofadr[pylon_z_joint]


# Bodies
pylon_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pylon"
)
foot_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "foot"
)

# Sensors
def get_sensor_adr(model, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return model.sensor_adr[sensor_id]

adr_pos    = get_sensor_adr(model, "esr_slide_pos")
adr_vel    = get_sensor_adr(model, "esr_slide_vel")
adr_force  = get_sensor_adr(model, "ankle_force")
adr_torque = get_sensor_adr(model, "ankle_torque")

print(f"k = {k_lin:.1f} N/m")
print(f"Joint range: {model.jnt_range[esr_slide_joint]}")


# ============================================================
# INITIAL CONDITIONS
# ============================================================
def reset_initial_conditions(data):
    mujoco.mj_resetData(model, data)
    data.qpos[esr_slide_qpos]  = z0
    mujoco.mj_forward(model, data)

reset_initial_conditions(data)

# ============================================================
# LOGGING
# ============================================================
n_steps = int(sim_time / dt)

# Initialization    
time_log      = np.zeros(n_steps)     # time                   [s]
z_log         = np.zeros(n_steps)     # ESR joint displacement [m]
zdot_log      = np.zeros(n_steps)     # ESR joint velocity     [m/s]
esr_force_log = np.zeros(n_steps)     # applied ESR force      [N]
grf_log       = np.zeros(n_steps)     # ground reaction force  [N]
contact_log   = np.zeros(n_steps)     # number of contacts
pylon_z_log   = np.zeros(n_steps)     # pylon z-position       [m]
foot_z_log    = np.zeros(n_steps)     # foot z-position        [m]

# Sensor logs
sensor_pos_log    = np.zeros(n_steps)      # esr_slide_pos
sensor_vel_log    = np.zeros(n_steps)      # esr_slide_vel
sensor_force_log  = np.zeros((n_steps,3))  # ankle_force  [Fx, Fy, Fz]
sensor_torque_log = np.zeros((n_steps,3))  # ankle_torque [Tx, Ty, Tz]


# ============================================================
# SIMULATION
# ============================================================
print(f"\nRunning ESR slide model")
 
for i in range(n_steps):
 
    # --- ESR force law (Linear Kelvin-Voigt Model) ---
    z     = data.qpos[esr_slide_qpos]
    z_dot = data.qvel[esr_slide_dof]
    Fz    = - k_lin * z - c_lin * z_dot    # F_z = -k_lin*z
    # Apply force in joint
    data.qfrc_applied[:]             = 0
    data.qfrc_applied[esr_slide_dof] = Fz
 
    # --- Step ---
    mujoco.mj_step(model, data)
 
    # --- GRF from contacts ---
    grf_z = 0.0
    for c in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, force)
        grf_z += force[0]
 
    # --- Log ---
    time_log[i]      = data.time
    z_log[i]         = z
    zdot_log[i]      = z_dot
    esr_force_log[i] = Fz + data.qfrc_passive[esr_slide_dof]
    grf_log[i]       = grf_z
    contact_log[i]   = data.ncon
    pylon_z_log[i]   = data.xpos[pylon_body, 2]
    foot_z_log[i]    = data.xpos[foot_body, 2]
 
    # --- Sensor readout ---
    sensor_pos_log[i]      = data.sensordata[adr_pos]
    sensor_vel_log[i]      = data.sensordata[adr_vel]
    sensor_force_log[i]    = data.sensordata[adr_force:adr_force+3]
    sensor_torque_log[i]   = data.sensordata[adr_torque:adr_torque+3]
 
print("Done")
 
# ============================================================
# PLOTS
# ============================================================
 
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle(
    f"ESR Slide Joint\n"
    f"k={k_lin:.0f} N/m  (Vari-Flex XC)",
    fontsize=11
)
 
# Body positions
axes[0, 0].plot(time_log, pylon_z_log, label="pylon")
axes[0, 0].plot(time_log, foot_z_log,  label="foot")
axes[0, 0].set_title("Body positions")
axes[0, 0].set_xlabel("Time [s]")
axes[0, 0].set_ylabel("Position [m]")
axes[0, 0].legend()
axes[0, 0].grid()
 
# ESR deformation
axes[0, 1].plot(time_log, z_log * 1000)
axes[0, 1].set_title("ESR deformation (slide joint)")
axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].set_ylabel("Displacement z [mm]")
axes[0, 1].grid()
 
# ESR force
axes[1, 0].plot(time_log, esr_force_log, label="ESR F_z (applied)")
axes[1, 0].set_title("ESR force")
axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].set_ylabel("Force [N]")
axes[1, 0].legend()
axes[1, 0].grid()
 
# GRF
axes[1, 1].plot(time_log, grf_log)
axes[1, 1].set_title("Ground reaction force")
axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].set_ylabel("GRF [N]")
axes[1, 1].grid()
 
# Force-Deformation
axes[2, 0].plot(z_log * 1000, esr_force_log)
axes[2, 0].set_title("Force-Deformation")
axes[2, 0].set_xlabel("Displacement z [mm]")
axes[2, 0].set_ylabel("ESR Force [N]")
axes[2, 0].grid()
 
# ESR vs GRF
axes[2, 1].plot(time_log, esr_force_log, label="ESR force")
axes[2, 1].plot(time_log, grf_log,       label="GRF")
axes[2, 1].set_title("ESR force vs GRF")
axes[2, 1].set_xlabel("Time [s]")
axes[2, 1].set_ylabel("Force [N]")
axes[2, 1].legend()
axes[2, 1].grid()
 
plt.tight_layout()
plt.savefig("esr_slide.png", dpi=150)
print("Saved: esr_slide.png")


# ============================================================
# INTERACTIVE VISUALIZATION
# ============================================================
print("Start interactive Simulation...")

# Reset for start
mujoco.mj_resetData(model, data)

data.qvel[esr_slide_qpos] = z0
viewer = mujoco.viewer.launch_passive(model, data)

# Camera onto model
viewer.cam.lookat[:] = [0, 0, 0.5]
viewer.cam.distance = 2.0
viewer.cam.azimuth = 45
viewer.cam.elevation = -20

while viewer.is_running():
    # Force computation and apply it
    z = data.qpos[esr_slide_qpos]
    z_dot = data.qvel[esr_slide_dof]
    Fz = -k_lin * z -c_lin * z_dot

    data.qfrc_applied[:] = 0
    data.qfrc_applied[esr_slide_dof] = Fz
    
    mujoco.mj_step(model, data)
    
    # Synchronize Viewer
    viewer.sync()
    
    # Pause to see it slower
    time.sleep(0.005)
