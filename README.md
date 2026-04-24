This repository contains the dissertation and associated code for:
**"Investigating Structured Reasoning Capabilities in Small-Scale Language Models Trained with Group Relative Policy Optimization for High School Physics Problem Solving"**

## Dissertation PDF
Click below to view the full paper:

[![View PDF](https://img.shields.io/badge/Open%20PDF-blue?style=for-the-badge&logo=adobeacrobatreader)](./dissertation.pdf)

## Code Archive
Download the full implementation:
[code](./code)

## Small-Model Reasoning for High School Physics via GRPO + LoRA
Fine-tuned Qwen2.5-3B Instruct using LoRA and Group Relative Policy Optimization (GRPO) to enhance reasoning on high school physics problems—achieving high accuracy without a critic model.

## Overview
Developed a lightweight and efficient approach to improve LLM reasoning under limited compute.
  Base model: Qwen2.5-3B Instruct
  Fine-tuning: LoRA for memory efficiency
  Training: GRPO (no critic model)
  Reward design: Structured reasoning + multi-dimensional evaluation

## Results
Evaluated on 199 physics problems (dialogue format):

Accuracy: 45% → 82%

Multiple-choice: 45% → 85%

Open-ended: 46% → 76%

Format adherence: ~98%

## Key Contributions
Designed reward function combining reasoning structure and answer correctness

Improved numerical reasoning using absolute percentage error (APE) penalty

Achieved strong performance with significantly lower resource requirements
