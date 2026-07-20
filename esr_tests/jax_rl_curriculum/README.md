## Curriculum Learning: From Mimicry to Adaptation

This example demonstrates training a PPO agent on the MjxSkeletonMuscleProsthesis environment using a two-phase curriculum. This approach ensures the agent first learns the fundamentals of human gait before adapting its strategy to the specific dynamics of the prosthetic hardware.

The framework leverages GoalTrajMimic to encode target trajectories (positions, orientations, and velocities) from expert datasets. The ProsthesisReward function facilitates this transition:
- Phase 1 (Mimicry): Employs a DeepMimic-style reward to align the agent with healthy baseline walking data.
- Phase 2 (Adaptation): Transitions to a task-based reward, allowing the agent to deviate from symmetric biological gait to find an optimal, stable locomotion strategy for the prosthetic configuration.

---

### 🚀 Training

To initiate the curriculum training, execute: 

```bash
sh experiment_curriculum.sh 
```

This command will:

- Two-Phase Training: The agent sequentially masters baseline gait (Phase 1) and prosthetic adaptation (Phase 2).
- Trained policies for both phases are saved as PPOJax_saved.pkl in the outputs/ directory. 
- Perform a final rendering of the both trained policies.
- Save a video of the rendering of both policies to the `LocoMuJoCo_recordings/` directory.
- Upload the video to Weights & Biases (WandB) for further analysis.


#### Validation Loop During Training

Throughout the training cycle, the policy is rigorously evaluated against the expert dataset using:
- Temporal Metrics: Dynamic Time Warping (DTW) and Discrete Fréchet Distance.
- Spatial Metrics: Euclidean distance across joint positions, velocities, and site orientations.
- Logging: All biomechanical fidelity metrics are logged directly to WandB.

---

### 📈 Evaluation

To evaluate the trained agent, run:

```bash
python eval.py --path path/to/agent_file
```

If you'd like to evaluate the agent using MuJoCo (instead of Mjx), run:

```bash
python eval.py --path path/to/agent_file --use_mujoco
```


