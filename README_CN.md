![CORY LOGO](img/CORY-LOGO.jpg)

# 与另一个你共同进化（CORY）

[English](README.md) | [中文](README_CN.md)

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue)](https://nips.cc/Conferences/2024)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06101-red)](https://arxiv.org/abs/2410.06101)
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)

---

## 🏛️ 关于本仓库

本仓库由 **[CASIA-Collect-AI](https://github.com/CASIA-Collect-AI)** 收录维护，作为 MARL 与大语言模型交叉领域高质量研究代码的精选集合。

📌 **原始仓库（推荐访问）：** [Harry67Hu/CORY](https://github.com/Harry67Hu/CORY)
⭐ **如果本工作对你有帮助，请前往原始仓库点 Star 支持作者！**

🔗 **MARL 框架：** [HMAP/HMP2G](https://github.com/binary-husky/hmp2g) — 配套的多智能体强化学习实验框架。

> **团队：** 中国科学院自动化研究所 飞行器智能技术团队（群体智能团队-蒲志强）
> CASIA-Collect-AI 收录和维护 MARL、LLM 和机器人领域的高质量开源研究代码。

---

**《与另一个你共同进化：用序列协作多智能体强化学习微调大语言模型》** 官方实现（NeurIPS 2024）

**作者：** 马昊\*（共同第一），胡天一\*（共同第一），蒲志强，刘博印，艾晓林，梁艳艳，陈敏
**单位：** 中国科学院大学；中国科学院自动化研究所；阿里巴巴（中国）有限公司；澳门科技大学

---

## 摘要

强化学习已成为在特定任务上微调大语言模型的关键技术。然而，现有方法主要依赖 PPO 及其变体，在应用于 LLM 微调时，往往表现出次优性能和对**分布崩溃**的脆弱性。

我们提出 **CORY**，将 LLM 的 RL 微调扩展到序列协作多智能体强化学习框架，利用多智能体系统固有的共同进化和涌现能力。CORY 将 LLM 拆分为两个智能体——**先锋体**和**观察体**——并以协作序列方式训练它们。该框架同时提升任务性能并缓解分布崩溃问题。

---

## 📖 论文深度解读

### 核心问题：PPO 微调 LLM 的两大瓶颈

用强化学习微调大语言模型（如 RLHF）已成为主流技术路线，但 PPO 在 LLM 微调中面临两个核心问题：

1. **策略最优性不足**：单智能体 PPO 容易陷入局部最优，尤其在奖励稀疏或非凸时；
2. **分布崩溃（Distribution Collapse）**：模型在追求高奖励的过程中，会偏离预训练分布，生成内容退化（如重复、语义漂移），KL 散度急剧上升。

**核心直觉**：如果两个共同进化的智能体相互学习、彼此约束，是否能规避单智能体的这些缺陷？

---

### CORY 方法详解

#### 双智能体设计：先锋体（Pioneer）与观察体（Observer）

LLM 被复制为两个自主智能体：

| 角色 | 输入 | 任务 |
|------|------|------|
| **先锋体（Pioneer）** | 查询 `q` | 直接生成回答 `a₁` |
| **观察体（Observer）** | 查询 `q` + 先锋体回答 `a₁` | 基于先锋体的参考生成回答 `a₂` |

观察体能看到先锋体的输出，本质上是在"学习如何改进同伴的答案"，形成知识传递链路。

![CORY 框架](imgs/fig_framework.png)
*CORY 框架总览：先锋体生成初始回答，观察体基于先锋体的输出进行改进*

#### 协作奖励：共同优化总体表现

两个智能体共享一个**集体奖励**：

```
r_CORY(s₀, a₁, a₂) = r(s₀, a₁) + r(s₀, a₂)
```

这意味着先锋体生成差的回答不仅影响自己，也会影响观察体的奖励——促使两者真正协同。

#### 角色周期性交换（Role Exchange）

每 `T_exchange` 步，先锋体和观察体**互换角色**：原来的观察体变成新的先锋体。这种交叉训练确保：
- 两个智能体都从两种生成模式中受益（独立生成 + 参考条件生成）
- 两者都不会过度专注于某一单一角色
- 相互约束防止分布崩溃

**为什么角色交换能防止分布崩溃：** 当观察体始终能看到先锋体的参考时，有了天然的"锚点"，防止偏离训练分布。先锋体知道自己的输出会被用作参考，也有动力保持连贯性。

---

### 实验结果

#### 情感微调（IMDB）

任务：微调 LLM 以生成正面电影评论。

![IMDB 结果](imgs/fig_imdb_results.png)
*IMDB 情感任务上的奖励曲线。与 PPO 基线相比，CORY 以更小方差实现更高的最终奖励*

**核心发现：**
- CORY 比 PPO、NLPO 和 TRPO 基线达到显著更高的正面情感分数
- CORY 中的分布崩溃（通过 KL 散度峰值衡量）大幅减少

#### 算术推理（GSM8K）

任务：微调 LLM 用于小学数学题求解。

![GSM8K 结果](imgs/fig_gsm8k_results.png)
*GSM8K 算术推理基准测试准确率。CORY 持续优于单智能体 RL 方法*

**核心发现：**
- CORY 比最佳 PPO 基线提升 GSM8K 准确率 3–5%
- 观察体访问先锋体推理链的过程起到隐式思维链增强的作用

#### 消融实验

![消融实验](imgs/fig_ablation.png)
*消融实验展示各组件贡献：协作奖励、角色交换和双智能体架构*

**消融发现：**
- 去掉角色交换导致性能退化到接近 PPO 水平
- 去掉协作奖励消除了相互约束，导致分布崩溃
- 两个组件都是必要且共同充分的

---

## 安装

```bash
conda create --name cory python=3.10
conda activate cory
pip install -r requirements.txt
```

**依赖：** PyTorch 2.2+, Transformers 4.37+, trl, peft, accelerate

---

## 快速开始

### 情感微调（IMDB）

```bash
python train.py \
  --task imdb \
  --model_name gpt2 \
  --method cory \
  --exchange_steps 100
```

### 算术推理（GSM8K）

```bash
python train.py \
  --task gsm8k \
  --model_name meta-llama/Llama-2-7b-hf \
  --method cory \
  --exchange_steps 200
```

---

## 引用

```bibtex
@inproceedings{ma2024cory,
  title={Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning},
  author={Ma, Hao and Hu, Tianyi and Pu, Zhiqiang and Liu, Boyin and Ai, Xiaolin and Liang, Yanyan and Chen, Min},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```

---

## 联系方式

- **共同第一作者：** 马昊，胡天一 — hutianyi2021@ia.ac.cn
- **通讯作者：** zhiqiang.pu@ia.ac.cn（蒲志强教授）
