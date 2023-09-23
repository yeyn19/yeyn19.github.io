---
title: '论文阅读[粗读]-Minimum Bayes-Risk Decoding for Statistical Machine Translation'
tags:
  - 人工智能
  - 计算机
categories: 论文阅读笔记
abbrlink: f1fcbd7
date: 2022-07-09 16:01:24
---

这篇文章是讲机器翻译的，准确来说，是解码策略。作者提到已有很多评测machine translation的方法，但已有的解码大多没有同时考虑这些方法。

本文提出的MBR方法可以：

- 根据特定的metric，解码出得分更高的候选
- 通过特定的loss函数，把语法树考虑进统计语言模型

