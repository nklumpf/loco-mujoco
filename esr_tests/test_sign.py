import numpy as np
import mujoco
import jax.numpy as jnp
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf

# Load trained PPO agent
agent_conf, _ = PPOJax.load_agent(
    "jax_rl_curriculum/outputs/2026-07-20/19-30-25/PPOJax_saved.pkl"
)
config = agent_conf.config

# Build environment
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"

env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    domain_randomization_type=config.randomization_config["randomization_type"],
    domain_randomization_params=config.randomization_config["randomization_params"],
)

# Access MuJoCo model and ESR hinge
model    = env._model
hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "esr_hinge_r")
hinge_qpos = model.jnt_qposadr[hinge_id]
hinge_dof  = model.jnt_dofadr[hinge_id]
L          = env.ESR_lever_arm

def compute_tau_numpy(theta):
    """Replicate compute_ESR_hinge_qfrc in pure numpy for testing."""
    z = L * np.sin(theta)
    k = env.linear_params["k_heel"] if theta < 0 else env.linear_params["k_keel"]
    F_z = k * z
    return -F_z * L * np.cos(theta)

print("=== TEST ON REAL MUJOCO MODEL (Forward + Step) ===")
for theta_deg in [-5.0, +10.0]:
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)

    # Set hinge angle
    d.qpos[hinge_qpos] = np.deg2rad(theta_deg)
    mujoco.mj_forward(model, d)

    # Compute ESR torque and apply it
    tau = compute_tau_numpy(np.deg2rad(theta_deg))
    d.qfrc_applied[hinge_dof] = tau

    # Step simulation
    mujoco.mj_step(model, d)

    theta_after = np.rad2deg(d.qpos[hinge_qpos])
    moved_toward_zero = abs(theta_after) < abs(theta_deg)

    print(f"\n  theta={theta_deg:+.0f}°  tau={tau:.2f} Nm")
    print(f"  After 1 step: {theta_after:.3f}°")
    print(f"  Moves toward zero: {'✓' if moved_toward_zero else '✗ WRONG'}")