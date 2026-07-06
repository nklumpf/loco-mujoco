import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf

# Defines the prosthesis configuration and biomechanical parameters of the simulated subject
env_params = {
    # Only required parameters, rest will be defaulted
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial", 
    "prosthesis_subtype": "ESR",
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2, 
}
# # example --> you can add as many datasets as you want in the lists!
# Create the locomotion simulation environment.
env = ImitationFactory.make("MjxSkeletonMuscleProsthesis", **env_params,
                            default_dataset_conf=DefaultDatasetConf(["walk"]),
                            # Select a musculoskeletal prosthetic human model

                            # lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]),
                            # if SMPL and AMASS are installed, you can use the following:
                            # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
                            #                                     "KIT/12/WalkInClockwiseCircle11_poses",
                            #                                     "HUMAN4D/HUMAN4D/Subject3_Medhi/INF_JumpingJack_S3_01_poses",
                            #                                     'KIT/359/walking_fast05_poses']),
                            n_substeps=20) # Use 20 physics simulation substeps for smoother and more stable simulation.

# Play and render the walking trajectories in the simulator: 3 separate walking demonstrations, each episode contains 500 simulation steps, open a visualization window showing the walking motion
env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)