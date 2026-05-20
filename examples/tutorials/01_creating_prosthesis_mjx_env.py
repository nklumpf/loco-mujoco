import os
import jax
import time

from loco_mujoco import ImitationFactory


# can increase the speed by ~30% on some GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# amputation and prosthesis parameters 
env_params = {
    # Amputation parameters
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial", 
    "amputated_tibia_length": 0.2,
    "reattach_muscles": {"med_gas": [0,0,0]}, # reconnect muscels of the residual limb after amputation

    # Prosthesis parameters
    "prosthesis_subtype": "SACH",
    "tibia_socket_overlap": 0.2, 
    "remove_joint_names": ["subtalar_angle"],
    "joint_stiffness": {"ankle_angle": 900}, # Automatically for prosthesis side 
    "adapt_joint_range": {"ankle_angle_r": [-10,10]},
    "socket_joint_dofs": ['socket_tx', 'socket_ty', 'socket_tz', 'socket_flexion', 'socket_adduction', 'socket_rotation'],
    "socket_joint_stiffnesses": {
        "socket_tx": 43500,
        "socket_ty": 43500,
        "socket_tz": 20000,
        "socket_flexion": 997,
        "socket_adduction": 623,
        "socket_rotation": 10
    },
    "socket_joint_dampings": {
        "socket_tx": 40,
        "socket_ty": 4,
        "socket_tz": 40,
        "socket_flexion": 10,
        "socket_adduction": 6,
        "socket_rotation": 2
    },
    "socket_ty_slack": True, 

    # contact parameters
    "multi_contact_geom_type": "2boxes",
    "contact_geom_solref":  [-900,-300],
    
    # for evaluation 
    "add_sensors": True,    # add biomechanical sensors for evaluation and analysis

}

# create env
env = ImitationFactory.make("MjxSkeletonMuscleProsthesis", **env_params, default_dataset_conf=dict(task="walk"))

# create keys
key = jax.random.key(0)
n_envs = 100                                # 100 environments simultaneously in parallel
keys = jax.random.split(key, n_envs + 1)    
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed         
# vmap: vectorizes simulation functions across all environments
# jit: Compiles functions for high-performance GPU execution
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# reset env
state = rng_reset(env_keys)

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
i = 0
while i < 100000:

    # step
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)

    # parallel render
    env.mjx_render(state)   # render simulation visually

    step += n_envs

    # log speed (disable rendering for accurate speed measurement)
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

    i+=1