# Reproducing π₀.₅ + LeRobot + LIBERO + MuJoCo on macOS

> This article lives at https://github.com/EmbodiedFX/LeRobot-Mac. 中文版: [README-CH.md](./README-CH.md)

This note documents how to get the following two things working from scratch on a **clean macOS Apple Silicon machine**:

1. **Single-step inference**: let π₀.₅ read one LIBERO data sample and output an action prediction
2. **Closed-loop evaluation**: let π₀.₅ act as the policy, actually enter the LIBERO environment, complete tasks in MuJoCo, and produce evaluation results

Once finished, you should get three kinds of outputs:

* **Single-step inference result**: `pred_action` printed in the console
* **Evaluation results**: success-rate statistics for test suites such as `libero_object` and `libero_spatial`
* **Simulation artifacts**: logs, configs, and generated simulation videos or image sequences in the evaluation output directory, for example:

https://github.com/user-attachments/assets/c750d724-570b-455f-a86f-a2c561a45fe5

Reference for this document:
- [LeRobot installation](https://huggingface.co/docs/lerobot/en/installation)
- [Using Libero](https://huggingface.co/docs/lerobot/en/libero)
- [π₀.₅ Policy](https://huggingface.co/docs/lerobot/en/pi05)
- [Environment Processors](https://huggingface.co/docs/lerobot/en/env_processor)

## 1. Background

What are the components mentioned in this article?

- **MuJoCo** is the underlying physics engine responsible for simulation.
- **LIBERO** is a robot manipulation benchmark, which you can think of as a collection of standard tasks.
- **LeRobot** is the toolchain that unifies models, datasets, and evaluation entry points.
- **π₀.₅** is the VLA policy used in this guide.

So this workflow is doing the following: load π₀.₅ with LeRobot, place it into the LIBERO environment, and run a real closed-loop evaluation on MuJoCo.

## 2. Target Environment

This article assumes the following environment:

* **macOS**
* **Apple Silicon (M3 Pro)**
* **conda / miniforge**
* Running locally, without relying on a remote Linux machine

## 3. Create a Python 3.12 Environment

If conda is not installed yet, you can install Miniforge first:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

After installation, reopen the terminal and create the environment:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg
python -m pip install -U pip
```

## 4. Install LeRobot from Source

There are two paths: installing via pip or from source. Installing from source is recommended, since it makes it easier to inspect the code or apply local modifications.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

## 5. Install π₀.₅ Dependencies

```bash
pip install -e ".[pi]"
```

## 6. Install LIBERO Runtime Dependencies

Install `hf-libero` directly:

```bash
pip install "hf-libero>=0.1.3,<0.2.0"
```

> Do not run `pip install -e ".[libero]"`. It pulls in the standard `transformers>=4.57.1`, which conflicts with π₀.₅’s dependency (a patched `transformers` branch: `fix/lerobot_openpi`).

### Troubleshooting

If you encounter a `cmake`-related build error during installation, try downgrading `cmake` to below 4:

```bash
python -m pip uninstall -y cmake
conda install -y -c conda-forge "cmake<4"
```

Then reinstall the relevant dependencies.

## 7. Verify That MPS and MuJoCo Are Available

At this point, the environment should be set up. You can first verify that both the device backend and the MuJoCo Python package are working correctly.

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())

import mujoco
print("mujoco:", mujoco.__version__)
PY
```

Expected output:

* `mps available: True`
* `mujoco` prints its version successfully

If you see that, it means:

* PyTorch can use MPS on the current Apple Silicon machine
* The MuJoCo Python bindings were installed successfully

## 8. Single-Step Inference

This step is to verify whether **π₀.₅ can successfully complete one forward inference pass in the current environment**.

Create a file named `quick_infer_mac.py` in the `lerobot/` root directory with the following contents:

```python
import torch

def _no_compile(fn=None, *args, **kwargs):
    return fn

torch.compile = _no_compile

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

model_id = "lerobot/pi05_libero_finetuned"

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

policy = PI05Policy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

dataset = LeRobotDataset("lerobot/libero")

episode_index = 0
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
frame = dict(dataset[from_idx])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    pred_action = postprocess(pred_action)

print("device:", device)
print("pred_action:", pred_action)
```

This code does 6 things:

1. Loads the π₀.₅ model `lerobot/pi05_libero_finetuned`
2. Places the model on `mps` or `cpu` depending on device availability
3. Fetches one sample from the `lerobot/libero` dataset
4. Uses `preprocess` to convert the sample into model-ready input
5. Calls `policy.select_action(...)` to predict the next action
6. Prints the predicted action

> This follows the same quick-start logic as the π₀.₅ model card.

Run:

```bash
python quick_infer_mac.py
```

If successful, you should see output like:

```text
device: mps
pred_action: ...
```

This means:

* The model loads successfully
* The dataset is accessible
* The processor works correctly
* MPS can perform at least one real inference pass

At this point, you have completed model-level validation.

## 9. Minimal Closed-Loop Evaluation

Next comes system-level validation: put π₀.₅ into the LIBERO / MuJoCo environment and run a rollout.

Start with the minimal version:

```bash
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.device=mps \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

Here is what each argument does:

* `--policy.path=lerobot/pi05_libero_finetuned`
  Specifies the π₀.₅ checkpoint to use

* `--policy.device=mps`
  Explicitly tells LeRobot to use MPS on Apple Silicon

* `--policy.compile_model=false`
  Disables compile to ensure evaluation does not enter an unstable compilation path

* `--policy.gradient_checkpointing=false`
  Turns off the memory-saving switch commonly used in training so that the inference path is more direct

* `--env.type=libero`
  Sets the environment type to LIBERO

* `--env.task=libero_object`
  Starts with one of the most commonly used and smallest suites for validation

* `--eval.batch_size=1`
  Launches only one environment instance to reduce resource usage

* `--eval.n_episodes=1`
  Runs just 1 episode first to confirm that the full closed-loop pipeline works

> The first time you run `lerobot-eval`, you may see:
> `Do you want to specify a custom path for the dataset folder? (Y/N):`
> If your goal is simply to get things running first, just type `N` and press Enter. This means using the default cache directory instead of manually specifying a download location.

Once the first episode runs successfully, you can increase the same suite to 3 episodes to verify that the result is not a one-off success.

```bash
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.device=mps \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=3
```

After finishing `libero_object`, you can try a different type of suite, such as `libero_spatial`:

```bash
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.device=mps \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=3
```

> The LIBERO documentation lists multiple suites, such as `libero_object`, `libero_spatial`, `libero_goal`, and `libero_10`. You can try them all.

Each time you run `lerobot-eval`, the terminal prints an output directory, for example:

```text
Output dir: outputs/eval/2026-03-16/...
```

This directory is the entry point for the results of the current evaluation. You can inspect the contents there:

* configuration files
* log files
* evaluation statistics
* mp4, gif, or image sequences generated when recording is enabled in the local version

The most satisfying output is usually the simulation video:
you can directly watch the robot arm execute the task in MuJoCo according to the actions produced by the policy.
