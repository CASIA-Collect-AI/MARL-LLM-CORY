![CORY LOGO](img/CORY-LOGO.jpg)

# Coevolving with the Other You (CORY)

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue)](https://nips.cc/Conferences/2024)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06101-red)](https://arxiv.org/abs/2410.06101)
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)

---

## 🏛️ 关于本仓库 | About This Repository

本仓库由 **[CASIA-Collect-AI](https://github.com/CASIA-Collect-AI)** 收录维护，作为多智能体强化学习与大语言模型交叉领域的优质论文代码集合。

📌 **原始仓库（推荐访问）：** [Harry67Hu/CORY](https://github.com/Harry67Hu/CORY)
⭐ **如果本工作对你有帮助，请前往原始仓库点 Star 支持作者！**

> CASIA-Collect-AI 是中国科学院自动化研究所 AI 团队维护的开源代码收录平台，专注于收录和整理 MARL、LLM、机器人等领域的高质量研究代码。

---

Official implementation of **Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning** (NeurIPS 2024).

**Authors:** Hao Ma\*(Co-First), Tianyi Hu\*(Co-First), Zhiqiang Pu, Boyin Liu, Xiaolin Ai, Yanyan Liang, Min Chen
**Affiliations:** University of Chinese Academy of Sciences; Institute of Automation, Chinese Academy of Sciences; Alibaba (China) Co., Ltd.; Macau University of Science and Technology

---

## Abstract

RL has emerged as a pivotal technique for fine-tuning LLMs on specific tasks. However, prevailing methods predominantly rely on PPO and variants, which often exhibit suboptimal performance and vulnerability to **distribution collapse** when applied to LLM fine-tuning.

We propose **CORY**, extending RL fine-tuning of LLMs to a sequential cooperative multi-agent RL framework, leveraging the inherent coevolution and emergent capabilities of multi-agent systems.

<p align="center">
  <img src="img/CORY-idea.png" alt="CORY Idea" width="70%">
  <br>
  <b>Basic Idea of CORY</b>
</p>

---

## 📖 论文深度解读

### 研究背景：PPO 微调 LLM 的两大瓶颈

用强化学习微调大语言模型（如 RLHF）已成为主流技术路线，但 PPO 在 LLM 微调中面临两个核心问题：

1. **策略最优性不足**：单智能体 PPO 容易陷入局部最优，尤其在奖励稀疏或非凸时；
2. **Distribution Collapse（分布崩溃）**：模型在追求高奖励的过程中，会偏离预训练分布，生成内容退化（如重复、语义漂移），KL 散度急剧上升。

**核心直觉**：如果两个共同进化的智能体相互学习、彼此约束，是否能规避单智能体的这些缺陷？

---

### CORY 方法详解

#### 双智能体设计：Pioneer（先锋体）与 Observer（观察体）

LLM 被复制为两个自主智能体：

| 角色 | 输入 | 任务 |
|------|------|------|
| **Pioneer（先锋体）** | 查询 `q` | 直接生成回答 `a₁` |
| **Observer（观察体）** | 查询 `q` + 先锋体回答 `a₁` | 基于先锋体的参考生成回答 `a₂` |

Observer 能看到 Pioneer 的输出，本质上是在"学习如何改进同伴的答案"，形成知识传递链路。

#### 协作奖励：共同优化总体表现

两个智能体共享一个**集体奖励**：

```
r_CORY(s₀, a₁, a₂) = r(s₀, a₁) + r(s₀, a₂)
```

这意味着 Pioneer 生成差的回答不仅影响自己，也会影响 Observer 的奖励——促使两者真正协同。

#### 角色周期性交换（Role Exchange）

每隔 **5 次迭代**，Pioneer 和 Observer 互换角色。这一设计的关键价值：

- **打破对称性陷阱**：防止两个智能体分化为"懒惰的追随者"和"过于激进的开拓者"
- **双向知识迁移**：每个智能体既学习如何"引领"也学习如何"改进他人"
- **Ablation 验证**：去掉角色交换后，KL 散度控制能力显著下降

---

### 实验结果分析

#### IMDB Review（主观奖励：情感正向性）

| 指标 | PPO | CORY |
|------|-----|------|
| Combined Reward 趋势 | 先升后**下降** | 持续**上升** |
| KL Divergence（训练结束） | > CORY 的 **2 倍** | 显著更低 |

PPO 的 combined reward 曲线在达到峰值后下降，正是 distribution collapse 的典型特征——模型牺牲了语言质量来追求情感奖励。

#### GSM8K（客观奖励：数学正确率）

- 基础模型：4-bit Llama-2-7b-chat
- **CORY pass@1 达到 18%**，而 PPO 的任务奖励在约 50 次迭代后达到峰值随即崩溃
- CORY 维持显著更低的 KL 散度，数学推理能力提升的同时保留了语言模型能力

#### Ablation Study 关键结论

去掉**任一**组件均导致性能下降：
- 无知识迁移（Observer 不看 Pioneer 输出）→ 失去协同增益
- 无角色交换 → KL 散度控制变差，接近 PPO 水平

---

### 与主流 LLM 训练方法的定位

| 方法 | 类型 | 特点 |
|------|------|------|
| **PPO** | 单智能体 RL | 基准方法，易 distribution collapse |
| **DPO** | 离线偏好优化 | 无需 RL 训练循环，但缺乏在线探索 |
| **GRPO** | 组相对策略优化 | DeepSeek-R1 使用，无需 Critic |
| **CORY** | 多智能体协作 RL | 在线训练 + 协同进化，抗分布崩溃 |

CORY 的核心优势是**在线训练的协同进化机制**，在需要持续改进且对分布漂移敏感的任务中尤为适合。

---

## Installation

```bash
conda env create -f trl_environment.yml
conda activate trl
```

---

## Quick Start

1. Download GPT-2 and distilbert-imdb models.

2. Fine-tune with PPO (baseline):
```bash
python imdb_train/ppo.py
```

3. Fine-tune with CORY:
```bash
python imdb_train/cory.py
```

---

## Repository Structure

| File / Directory | Description |
|------------------|-------------|
| `imdb_train/` | IMDB Review experiments: `ppo.py`, `cory.py` |
| `gsm8k_utils/` | GSM8K utilities *(removed in CASIA fork)* |
| `trl/` | TRL-based trainers: PPO, CORY, SFT, DPO *(removed in CASIA fork)* |
| `utils/` | Model generation, experiment tracking *(removed in CASIA fork)* |
| `img/` | Figures and logos |
| `trl_environment.yml` | Conda environment specification |

---

## Paper & Citation

📄 [Coevolving with the Other You (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf)

```bibtex
@inproceedings{NEURIPS2024_1c2b1c8f,
  author = {Ma, Hao and Hu, Tianyi and Pu, Zhiqiang and Liu, Boyin and Ai, Xiaolin and Liang, Yanyan and Chen, Min},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {15497--15525},
  title = {Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning},
  volume = {37},
  year = {2024}
}
```
