# DiTRL: An Open Source Library for Planning using Diffusion Models

## Motivation

Reinforcement Learning has been the go-to learning paradigm for trajectory planning, where the model predicts (state, action) pairs from start to goal state. Recent works, including Diffuser[1], have proposed modeling trajectory planning as a generation problem which can be controlled with classifier guidance. However, existing methods have three major limitations: 

1. The backbone used for training diffusion models is not suitable for sequential prediction tasks
2. Lack of open-source and easy-to-use library for training diffusion models for planning, and,
3. Lack of application in complex planning situations.

## Citations

* [[1](https://arxiv.org/pdf/2205.09991.pdf)] Planning with Diffusion for Flexible Behavior Synthesis 
* [[2](https://arxiv.org/pdf/2310.07842.pdf)] DiPPeR: Diffusion-based 2D Path Planner applied on Legged Robots 
* [[3](https://arxiv.org/pdf/2210.17366.pdf)] CTG 
* [[4](https://arxiv.org/pdf/2306.06344.pdf)] CTG++ 
* [[5](https://openreview.net/forum?id=hclEbdHida)] DiffScene 
* [[6](https://arxiv.org/pdf/1904.01201.pdf)] Habitat: A Platform for Embodied AI Research 
* [[7](https://arxiv.org/abs/2212.09748)] DiT: Scalable Diffusion Models with Transformers, William Peebles and Saining Xie
