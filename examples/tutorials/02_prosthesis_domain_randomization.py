import numpy as np
import jax
import jax.numpy as jnp

from loco_mujoco.core import ObservationType
from loco_mujoco import ImitationFactory

# Enables prosthesis-specific domain randomization.
randomization_type = "ProsthesisRandomizer" 

randomization_config = {
    "visualize_randomization": True,  # visualize the randomized parameters, including body position and orientation, in the rendered environment
    # stiffness
    "randomize_prosthesis_joint_stiffness": False, #True,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [10, 100]},

    # damping
    "randomize_prosthesis_dof_damping": False, #True,
    "prosthesis_dof_damping_range": {'ankle_angle': [0, 2]},

    # body position
    "randomize_prosthesis_body_position": True,
    # "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.02, 0.02], 'z': [-0.02, 0.02]}, 'talus': {'x': [-0.02, 0.02], 'z': [-0.02, 0.02]}},
    "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.4, -0.4]}}, #, 'talus': {'x': [0.1, 0.1]}}, # random alignment is fixed

    # body orientation
    "randomize_prosthesis_body_orientation": False, #True,
    "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-0.2, 0.2], 'y': [-0.3, 0.3], 'z': [-0.2, 0.2]}, 'talus': {'x': [-0.2, 0.2],'y': [-0.3, 0.3], 'z': [-0.2, 0.2]}},
}

# amputation and prosthesis parameters
env_params = {
    # Only required parameters, rest will be defaulted
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial", 
    "prosthesis_subtype": "SACH",
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2, 
    "visualize_prosthesis": True,
}


# create the environment and task
env = ImitationFactory.make("MjxSkeletonMuscleProsthesis", **env_params, default_dataset_conf=dict(task="walk"),
                            domain_randomization_type=randomization_type,
                            domain_randomization_params=randomization_config)

# create keys
key = jax.random.key(0)
n_envs = 1 #100
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space)) 
# sample_action_space: Generates random control actions.

# reset env
state = rng_reset(env_keys)

step = 0
i = 0
while i < 100000:

    # step
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)

    # parallel render
    env.mjx_render(state)

    step += n_envs

    i+=1
