---
title: '从Switch-Transformer到Switch-Head:聊聊MoE'
tags:
  - 人工智能
  - 计算机
  - 分布式训练
  - transformer结构探索
categories: 论文阅读笔记
photo: ../../files/images/MoE/switchTransformers.png
mathjax: true
abbrlink: 7d9021cb
date: 2024-01-11 22:40:52
---

好久不更新了，今天来聊聊MoE技术，一个很早就有的、一直到最近才开始陆陆续续出现的、对提升模型能力实际上很重要的技术。参考论文：

> Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
>
> SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention
>
> Mixtral of Experts
>
> MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts
>
> Adaptive Mixture of Local Expert
>
> A Review of Spaerse Expert Models in Deep Learning
>
> Unified Scaling Laws for Routed Language Models

<!-- more -->
