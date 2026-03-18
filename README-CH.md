# 在 macOS 上复现 π₀.₅ + LeRobot + LIBERO + MuJoCo

> 本文位于 https://github.com/EmbodiedFX/LeRobot-Mac 。English version: [README.md](./README.md)

这篇笔记记录的是：在一台**干净的 macOS Apple Silicon 机器**上，如何从零开始跑通下面两件事：

1. **单步推理**：让 π₀.₅ 读取一条 LIBERO 数据样本，输出一个动作预测
2. **闭环评测**：让 π₀.₅ 作为策略，真正进入 LIBERO 环境，在 MuJoCo 上完成任务，并产出评测结果

完成后，你可以得到三类结果：

* **单步推理结果**：控制台打印 `pred_action`
* **评测结果**：`libero_object`、`libero_spatial` 等测试套件的成功率统计
* **仿真产物**：评测输出目录中的日志、配置，以及生成的仿真视频或图像序列，如

https://github.com/user-attachments/assets/c750d724-570b-455f-a86f-a2c561a45fe5

**本文档参考：**

* [LeRobot 安装](https://huggingface.co/docs/lerobot/en/installation)
* [使用 Libero](https://huggingface.co/docs/lerobot/en/libero)
* [π₀.₅ Policy](https://huggingface.co/docs/lerobot/en/pi05)
* [环境处理器](https://huggingface.co/docs/lerobot/en/env_processor)

## 一、背景

本文提到的组件分别是什么：

- **MuJoCo** 是底层物理引擎，负责仿真。
- **LIBERO** 是机器人操作任务基准，可以理解为一组标准任务集合。
- **LeRobot** 是把模型、数据集、评测入口统一起来的工具链。
- **π₀.₅** 是本次使用的 VLA policy。

因此，这套流程是在做这么一件事：用 LeRobot 加载 π₀.₅，把它放进 LIBERO 环境里，在 MuJoCo 上完成一次真实的闭环评测。

## 二、适用环境

本文默认环境如下：

* **macOS**
* **Apple Silicon（M3 Pro）**
* **conda / miniforge**
* 本地直接运行，不依赖远程 Linux 机器

## 三、创建 Python 3.12 环境

如果本机还没有 conda，可以先安装 Miniforge：

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

安装完成后，重新打开终端，创建环境：

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg
python -m pip install -U pip
```

后续安装都在这个环境里完成。

## 四、安装 LeRobot 源码

有 pip 安装和从源码安装两种路径。建议直接使用源码安装，方便查看源码或做本地修改。

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

## 五、安装 π₀.₅ 依赖

```bash
pip install -e ".[pi]"
```

## 六、安装 LIBERO 运行依赖

直接安装 `hf-libero`：

```bash
pip install "hf-libero>=0.1.3,<0.2.0"
```

> 不要跑`pip install -e ".[libero]"`，它引入标准版 `transformers>=4.57.1`，会与 π₀.₅ 的依赖（一个 transformers 的修复分支`fix/lerobot_openpi`）冲突。

### Troubleshooting

如果安装过程中遇到 `cmake` 相关的编译错误，试试把 `cmake` 降到 4 以下：

```bash
python -m pip uninstall -y cmake
conda install -y -c conda-forge "cmake<4"
```

然后重新安装相关依赖。

## 七、检查 MPS 和 MuJoCo 是否可用

环境已经搭好了。可以先确认设备后端和 MuJoCo Python 包是否都正常。

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

期望输出：

* `mps available: True`
* `mujoco` 能正常打印版本号

看到的话，说明：

* PyTorch 在当前 Apple Silicon 机器上可以使用 MPS
* MuJoCo Python 绑定安装成功

## 八、单步推理

这一步是在验证 **π₀.₅ 能不能在当前环境里顺利完成一次前向推理。**

在 `lerobot/` 根目录新建文件 `quick_infer_mac.py`，内容如下：

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

这段代码做了 6 件事：

1. 加载 π₀.₅ 模型 `lerobot/pi05_libero_finetuned`
2. 根据设备情况，把模型放到 `mps` 或 `cpu`
3. 从 `lerobot/libero` 数据集中取一条样本
4. 通过 preprocess 把样本整理成模型接受的输入
5. 调用 `policy.select_action(...)`预测下一步动作
6. 打印模型预测出的动作

> 这和 π₀.₅ 模型卡里的 quick start 逻辑是一致的。

执行：

```bash
python quick_infer_mac.py
```

如果成功，会看到类似输出：

```text
device: mps
pred_action: ...
```

这意味着：

* 模型可以正常加载
* 数据集可以正常访问
* processor 可以正常工作
* MPS 可以完成至少一次真实推理

至此，你已经完成了模型级验证。

## 九、最小闭环评测

接下来进入系统级验证，把 π₀.₅ 放进 LIBERO / MuJoCo 环境里做 rollout。

先跑最小版本：

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

这里每个参数的作用如下：

* `--policy.path=lerobot/pi05_libero_finetuned`
  指定使用的 π₀.₅ checkpoint

* `--policy.device=mps`
  明确告诉 LeRobot 在 Apple Silicon 上使用 MPS

* `--policy.compile_model=false`
  禁用 compile，保证评测时不进入不稳定的编译路径

* `--policy.gradient_checkpointing=false`
  关闭训练场景常用的显存优化开关，让推理路径更直接

* `--env.type=libero`
  指定环境类型为 LIBERO

* `--env.task=libero_object`
  先选一个最常用、最小的 suite 做验证

* `--eval.batch_size=1`
  只开一个环境实例，减少资源占用

* `--eval.n_episodes=1`
  先只跑 1 个 episode，确认整条闭环链路是通的

> 第一次运行 `lerobot-eval` 时，可能会出现
`Do you want to specify a custom path for the dataset folder? (Y/N):`
如果目标只是先跑通，直接输入 `N` 并回车即可。这表示使用默认缓存目录，而不是手动指定下载位置。

当第一个 episode 跑通后，可以把同一个 suite 提升到 3 个 episode，验证结果不是偶然成功。

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

完成 `libero_object` 之后，可以试试不同类型的 suite，例如 `libero_spatial`：

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

> LIBERO 文档里列出了多个 suite，例如 `libero_object`、`libero_spatial`、`libero_goal`、`libero_10` 等，可以都试下。

每次运行 `lerobot-eval`，终端都会打印一个输出目录，例如：

```text
Output dir: outputs/eval/2026-03-16/...
```

这个目录就是本次评测的结果入口。可以查看这里的内容：

* 配置文件
* 日志文件
* 评测统计
* 本地版本启用录像时生成的 mp4、gif 或图像序列

最有获得感的结果通常就是这里的仿真视频：
可以直接看到机械臂在 MuJoCo 里按照 policy 输出的动作去执行任务。
