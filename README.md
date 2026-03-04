![CORY LOGO](img/CORY-LOGO.jpg)

# Coevolving with the Other You (CORY)

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue)](https://nips.cc/Conferences/2024)
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning** (NeurIPS 2024).

**Authors:** Hao Ma*(Co-First), Tianyi Hu*(Co-First), Zhiqiang Pu, Boyin Liu, Xiaolin Ai, Yanyan Liang, Min Chen  
**Affiliations:** University of Chinese Academy of Sciences; Institute of Automation, Chinese Academy of Sciences; Alibaba (China) Co., Ltd.; Macau University of Science and Technology

---

## Abstract

Reinforcement learning (RL) has emerged as a pivotal technique for fine-tuning large language models (LLMs) on specific tasks. However, prevailing RL fine-tuning methods predominantly rely on PPO and its variants. Though these algorithms are effective in general RL settings, they often exhibit suboptimal performance and vulnerability to distribution collapse when applied to the fine-tuning of LLMs.

In this paper, we propose **CORY**, extending the RL fine-tuning of LLMs to a sequential cooperative multi-agent reinforcement learning framework, to leverage the inherent coevolution and emergent capabilities of multi-agent systems.

- **Dual-agent design:** The LLM to be fine-tuned is duplicated into two autonomous agents: a *pioneer* and an *observer*. The pioneer generates responses based on queries; the observer generates responses using both the queries and the pioneer's responses.
- **Role exchange:** The two agents are trained together. During training, the agents exchange roles periodically, fostering cooperation and coevolution between them.
- **Experiments:** We evaluate CORY by fine-tuning GPT-2 and Llama-2 under subjective and objective reward functions on the IMDB Review and GSM8K datasets. Results show that CORY outperforms PPO in terms of policy optimality, resistance to distribution collapse, and training robustness.

<p align="center">
  <img src="img/CORY-idea.png" alt="CORY Idea" width="70%">
  <br>
  <b>Basic Idea of CORY</b>
</p>

---

## Installation

### Requirements

We recommend using a Conda virtual environment:

```bash
conda env create -f trl_environment.yml
conda activate trl
```

> **Note:** If you haven't installed Conda, please visit the [Anaconda website](https://www.anaconda.com/products/individual) to download and install it first.

---

## Quick Start

1. **Download models:** GPT-2 and distilbert-imdb (for IMDB experiments).

2. **Fine-tune with PPO (baseline):**

```bash
python imdb_train/ppo.py
```

3. **Fine-tune with CORY:**

```bash
python imdb_train/cory.py
```

---

## Repository Structure

| File / Directory | Description |
|------------------|-------------|
| `imdb_train/` | IMDB Review experiments: `ppo.py`, `cory.py` |
| `gsm8k_utils/` | GSM8K utilities: evaluation and data processing |
| `trl/` | TRL-based trainers: PPO, CORY, SFT, DPO, etc. |
| `utils/` | Model generation, text generation, experiment tracking |
| `img/` | Figures and logos |
| `trl_environment.yml` | Conda environment specification |

### Core Components

**1. Training Scripts (`imdb_train/`)**

- `ppo.py` — PPO baseline for LLM fine-tuning
- `cory.py` — CORY: sequential cooperative multi-agent RL fine-tuning

**2. TRL Framework (`trl/`)**

- `trainer/` — PPO trainer, reinforce trainer, SFT trainer, DPO trainer
- `models/` — Value head, base models
- `environment/` — Base environment for MARL

**3. Utilities (`utils/`, `gsm8k_utils/`)**

- Model generation, text generation, experiment tracking
- GSM8K evaluation and data processing

---

## Paper

[Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf) (NeurIPS 2024)

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{NEURIPS2024_1c2b1c8f,
  author = {Ma, Hao and Hu, Tianyi and Pu, Zhiqiang and Liu, Boyin and Ai, Xiaolin and Liang, Yanyan and Chen, Min},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {15497--15525},
  publisher = {Curran Associates, Inc.},
  title = {Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```

---

## Statement

We will continue to maintain this code repository in the coming months.

