# V-CoT: Hierarchical Visual Chain-of-Thought for Sensorimotor Control

This repository implements **V-CoT**, a hierarchical Vision-Language-Action (VLA) framework that bridges high-level semantic reasoning with low-level physical execution. By leveraging a Vision-Language Model (VLM) for visual subgoal generation and an Optimal Transport Flow Matching (OT-Flow) policy for precise visuomotor control, the system achieves robust closed-loop manipulation in complex environments.

## 1. Overview

Traditional end-to-end VLA models often struggle with long-horizon tasks due to a lack of intermediate temporal reasoning. **V-CoT** addresses this by decomposing the control loop into two distinct hierarchies:

* **High-Level Brain (VLM):** An `InstructPix2Pix` model fine-tuned via `LoRA` that generates a "Visual Chain-of-Thought"—predicting a visual subgoal (future state image) based on current observations and language instructions.
* **Low-Level Cerebellum (Flow Matching):** A conditional UNet-based controller that solves the Probability Flow ODE to execute high-frequency action chunks.

By transitioning from standard Diffusion to **Flow Matching**, the system achieves 10x faster inference and smoother trajectory generation via deterministic Euler integration.

---

## 2. Architecture and Data Flow

The architecture follows a multi-rate control scheme:

1. **Semantic Reasoning:** Every $N$ steps, the VLM generates a visual subgoal $\hat{s}_{subgoal}$ representing the state at $t+k$.
2. **Visuomotor Execution:** The low-level policy consumes the current observation $s_t$ and the visual subgoal $\hat{s}_{subgoal}$ to predict an action trajectory.
3. **Flow Matching Formulation:** The controller learns the velocity field $v_t$ that transports a Gaussian prior $x_0$ to the empirical action distribution $x_1$ following:

$$dx_t = v_t(x_t, t) dt$$



---

## 3. Key Features

* **Optimal Transport Flow Matching (OT-Flow):** Replaces traditional diffusion schedulers with linear interpolation trajectories for faster convergence and more efficient inference.
* **Visual Subgoal Conditioning:** Utilizes Classifier-Free Guidance (CFG) to align the robot's physical actions with the VLM's imagined future state.
* **Multi-Task Robosuite Integration:** Pre-configured for `Lift`, `PickPlaceCan`, and `NutAssemblySquare` tasks using the Panda robot arm.
* **LoRA Fine-tuning:** Efficient adaptation of high-level generative models to specific robotic domain datasets.

---

## 4. Installation

Ensure you have a CUDA-capable GPU and Python 3.8+.

```bash
git clone https://github.com/lloydkwak/V-CoT_Diffusion.git
cd V-CoT_Diffusion
pip install -r requirements.txt

```

*Dependencies include: PyTorch, Diffusers, PEFT, Robosuite, Hydra, and WandB.*

---

## 5. Pipeline Execution

### A. Data Extraction

Generate paired data (observation, subgoal, action) for V-CoT training:

```bash
python3 scripts/extract_vlm_dataset.py

```

### B. High-Level VLM Training

Fine-tune the subgoal generator using LoRA:

```bash
python3 scripts/train_vlm.py

```

### C. Low-Level Policy Training (Flow Matching)

Train the multitask visuomotor controller:

```bash
cd diffusion_policy
python3 train.py --config-name=train_diffusion_unet_image_workspace \
    task=multitask_image_vcot \
    hydra.run.dir=/workspace/data/outputs/flow_matching_multitask

```

---

## 6. Evaluation

### Independent Policy Testing (Teacher Forcing)

To isolate the performance of the Flow Matching controller using ground-truth subgoals:

```bash
python3 eval_lowlevel_policy.py

```

### Full System Deployment (Closed-Loop)

To evaluate the integrated VLM-Brain and FM-Cerebellum:

```bash
python3 scripts/eval_system_pipeline.py

```

---

## 7. Results and Visualization

Training and evaluation logs are integrated with **Weights & Biases (WandB)**. The evaluation scripts generate side-by-side videos of the live robot execution and the VLM's imagined subgoals to provide interpretability for the model's reasoning process.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{lloyd2026vcot,
  title={V-CoT: Hierarchical Visual Chain-of-Thought for Sensorimotor Control via Flow Matching},
  author={Kwak, Lloyd},
  journal={GitHub Repository},
  year={2026}
}

```

---
