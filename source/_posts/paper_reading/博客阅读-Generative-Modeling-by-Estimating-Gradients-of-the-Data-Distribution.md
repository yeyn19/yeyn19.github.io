---
title: 博客阅读-Generative Modeling by Estimating Gradients of the Data Distribution
tags:
  - 计算机
  - diffusion model
categories: 论文阅读笔记
description: diffusion model 看做一种 log score 近似，与里面的微分方程理论
mathjax: true
abbrlink: 593c50eb
date: 2022-06-26 16:38:48
---

主要是阅读了下面这篇博客：

> [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.github.io/blog/2021/score/)

这是去年ICLR论文作者对其论文的讲解

> SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS

生成模型都是想要通过训练集获取数据的分布(比如人脸数据集就是人脸长什么样)，再通过采样一个和源数据集分布相同的数据进行生成。

## 经典方法

作者谈到了经典生成模型里两种形式:

- **likelihood-based models**: 直接获取源数据集中数据的分布，或者用某种上下界来近似。auto-regressive， energe-based model, flow model, VAE都是这种情况
  - 问题是：需要保证分布是归一化的，因此很难大规模地应用在各种任务上（需要改模型结构）。或者，用某种上下界来估计分布，因此会有偏差。
- **implicit generative models**：隐式表示数据集中数据的分布。比如GAN，通过一个分类器的得分来表示和原数据集分布的区别，进而让模型来学习。
  - 问题是：需要对抗训练，很难收敛。生成的内容也比较容易崩溃。

## score function,  score-based model, score matching

我们定义这个score:
$$
score = \bigtriangledown_x \log p(x)
$$
好处在于，因为是对于导数的模拟，因此，这个score不需要归一化。不管怎么样都是一个归一的分布。score based model就是在逼近这个分布：
$$
s_\theta(x) \approx \bigtriangledown_x \log p(x)
$$

#### 如何逼近？

可以看出，score model是对概率分布的对数导数做估计，需要找一个满足这个条件的分布方法——Fisher-deivergence:
$$
\text{Fisher Distance} = E_{P(x)} || \bigtriangledown_x\log p(x) - s_\theta||^2_2
$$
这个可以衡量两个分布的距离：

- 越“接近“的分布距离越小
- 相同的分布距离是0

通过优化fisher距离，我们就能让score based model学到训练集的数据分布了。同时，通过一些score matching的算法，我们不需要知道原始数据集的$\bigtriangledown_x\log p(x)$就能优化，也不需要对抗的训练。

## Langevin dynamics

训练完成以后，如何采样呢？
$$
x_{i+1} = x_i + \epsilon \bigtriangledown_x\log p(x) + \sqrt{2\epsilon} z_i
$$

$$
z_i \in N(0,1), x_0 \in N(0,1),
$$

通过多部迭代，可以保证一个随机的点最终会以$p(x)$的概率存在在空间中

其中上面的导数部分用score based model近似。这样，我们不需要额外的计算就可以采样了



## score-based model 问题与解决

刚才的 fisher-rao distance存在的问题是：

- 由于使用2阶距离，对于本身p比较小的位置。训练时的导数过小，可能学不到。

也就是说，模型对于低密度区域的p(x)近似效果不好。这个问题实际的后果是：

- 在采样时需要从一个随机的点开始，然而最开始没落到高密度区域的话，模型很难让通过几步变化走到高密度区域。也就是说，开始随机到低密度区域(这是大概率事件，20%的区域享有80%的概率)，很可能会生成炸。

对于上面的问题，一个解决办法思路：

- 能不能对原始分布叠加上一些随机噪音。让高低密度区域不要区分的这么明显，进而让 score model 学到所有位置的分布

这个思路很好，但是有几个衍生的问题：

- "噪声"取多大？

有一个折中的办法：

- 多步加噪声，每次都是上一轮分布再叠加一个噪声。这样后面的分布趋向于随机分布。前面的分布和原始分布非常接近。

这种情况下，我们的score-based model 也需要相应的改变：
$$
S_\theta(x) \rightarrow S_\theta(x,t)
$$

$$
S_\theta(x,t) \approx \bigtriangledown_x\log p(x,t)
$$

可以逼近每一个采样步骤的分布。在学习这个score model时，我们需要考虑所有的噪声分布，加权：
$$
\sum_{i=1}^L \lambda_i E_{P(x,i)} || \bigtriangledown_x\log p(x,i) - S_\theta(x,i)||^2_2
$$
其中，权重一般就选取每一轮高斯噪声的方差：
$$
\lambda_i = \sigma_i^2
$$
同时，在采样的过程中，我们修改原始的Langevin dynamics采样:
$$
x_{i+1} = x_i + \epsilon S_\theta(x,T-i) + \sqrt{2\epsilon} z_i
$$
让每一步点都按照不同的分布进行行走。最终逐步去噪到一个原始分布的点。这个算法叫做**annealed Langevin dynamics**

博客后面关于微分方程的部分我打算在论文阅读完以后写上
