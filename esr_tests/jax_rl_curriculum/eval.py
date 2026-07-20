import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# headless
os.environ["MUJOCO_GL"] = "egl"


# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
domain_randomization_type = config.randomization_config["randomization_type"]
domain_randomization_params =  {
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {}, 
    "randomize_prosthesis_joint_stiffness": False, 
    "prosthesis_joint_stiffness_range": {}, 
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {"pylon_socket": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]}, "talus": {"x": [-0.0, 0.0], "z": [-0.0, 0.0]}},
    "randomize_prosthesis_body_orientation": True,
    "prosthesis_body_orientation_range": {
      "pylon_socket": {
        "x": [-0.0, 0.0],
        "y": [-0.0, 0.0],
        "z": [-0.0, 0.0]
      },
      "talus": {
        "x": [-0.0, 0.0],
        "z": [-0.0, 0.0]
      }
    },
}

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params,
                   domain_randomization_type=domain_randomization_type,
                    domain_randomization_params=domain_randomization_params)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=1000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                       train_state_seed=0)
