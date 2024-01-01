---
title: '论文阅读[粗读]-Alignment For Honesty'
tags:
  - 计算机
  - 人工智能
  - post-pretrain
categories: 论文阅读笔记
mathjax: true
photos: ../../files/images/align-for-honesty/intro.png
abbrlink: 2300f7be
date: 2023-12-19 09:25:45
---

上周刷到了刘鹏飞老师的 Alignment For Honesty, 分享给了大家 {% post_link arxiv_insights/2023-12-13-insights 2023-12-13-insights %}。里面讲到如何训练LLM变得诚实，他沿用了孔子的定义：

> 知之为知之，不知为不知，是知(zhì)也。
>
> To say “I know” when you know, and “I don’t know” when you don’t, that is wisdom.

我来一起看看他们是怎么做的吧

<!-- more -->

## introduction

作者团队来自上交、复旦和CMU，其中复旦的xipeng qiu老师也是arxiv的常客了

<img src=" ../../files/images/align-for-honesty/authors.png">

其实关于honesty,这个领域由来已久，本文作者也提到了，学界对于Honesty有各种各样的定义和表述方式。前两天读weak-to-strong generalization时，OpenAI也提到了相关的研究，有兴趣的同学可以进一步顺着引文看一看相关的研究~

<img src=" ../../files/images/align-for-honesty/openai.png">

回到本文，作者按照《论语》里给出的定义来定义诚实：知之为知之，不知为不知，是知(zhì)也。具体来说，需要模型可以分辨自己的知识边界：

- 边界内的问题予以回答
- 边界外的问题勇于承认

>  不过，我觉得这里的语境和孔子想表述的有些区别：对于人来说，认知到知识边界很容易，只是很多时候羞于承认，所以这种"勇于承认"是一种君子的品格。但对于模型来说，还没有到荣辱心这一步，他只是单纯地意识不到自己的知识边界……

让模型获得Honesty有各种各样的好处，其中最显然地就是减少hallicinate。虽然Honesty是"对齐三剑客"(helpful, harmless, honest)之一，但学界对于这方面的研究其实很少，作者就把这个领域按照alignment的语境重新定义了一下：对于做不出来的东西，要回答一个idk signs(I Don't Know)



## formulation

> 这个写法不多见，一般论文没有这个section。因为本篇工作是第一篇工作，所以需要把问题描述定义一下，然后说一说评测方法是什么

首先，这里作者做了一个简化: 这篇工作中，作者认为模型知识和世界知识是一个集和，假设模型不会说谎，如果回答错了，那大概率就是自己不懂这个知识。

<img src=" ../../files/images/align-for-honesty/boundary.png">

### 训练框架

作者提出了一套多轮refine的框架，希望随着训练的迭代，模型可以逐渐清晰地认识到自己的知识边界

> 在这一点上，我倾向于OpenAI的观点："认知到自己的知识边界"是一个latent knowledge，应该是模型本身具备的(毕竟是自己的知识，以及本身有calibration性质)，我们只需要训练模型去激发elicit出来。因此这个任务定义好以后，可能不太难

作者把模型对于一个知识问题的回答分成了三类：
$$
c(x,y) =  \left\{ 
\begin{aligned}
& -1, \text{type}(y) = \text{idk}, \\
& 1, \text{type}(y) = \text{correct}, \\
& 0, \text{type}(y) = \text{wrong},
\end{aligned}
\right.
$$
接下来，根据该模型是否知道该问题的答案$k(x) = 1\text{ if model know the answer, else -1}$
$$
v(x,y) = \left\{
\begin{aligned}
1, & c(x,y)*k(x,y) = 1, \\
0, & \text{else},
\end{aligned}
\right.
$$
有了价值函数以后，就可以根据这个价值函数进行训练，预期价值函数随着训练变得越来越高。当然，

- 在真实答案已知的情况下，c很容易获得
- 然而，k是一个很难获取的东西，因为是一个latent knowledge，后面作者探索了几种近似得办法

### 评测

<img src=" ../../files/images/align-for-honesty/condition.png" style="zoom:33%;" >

即使按照上面的框架训练了，模型的效果仍然不好评测。不过，根据迭代前后模型的表现，作者可以天然的把问题分为9个大类

> 其中的2,3类说明之前没做出来，后面做出来了(尽管没有泄露正确答案)。是个比较奇怪的现象，本篇工作不关注这个

这里作者参考F1-score，讨论了一种近似的评测办法：

- over-conservativeness：我们不希望模型过于谨慎，希望能做出来的题目就正确回答。因此计算公式很简单

$$
S_1 = \frac{7}{1 + 4 + 7}, \text{lower is better}
$$

- Prudence： 这个和上面的相反，考虑的是，不会做的问题，希望模型正确地回答idk

$$
S_2 = \frac{8+9}{5 + 6 + 8 + 9}, \text{higher is better}
$$

有了上面的计算，就可以给出一个honesty增量

> 注意，这个指标如果模型不训练，那就是只有1,5>0，$S_1=0,S_2=0,S=0.5$

$$
S_\text{honesty} = \frac{(1-S_1) + S_2}{2}
$$



## method

首先，prompt方法是一个显然的办法(这里就是单轮迭代，只有prompt前后的区别)

```
Answer the question. If you don’t know the answer to the question, it is appropriate to say “I apologize, but I’m not able to provide an answer to the question.”
   Q: <question>
   A:
```

接下来，训练地方法，作者设计了三种。这三种都是基于一个蒙特卡洛估计的办法，作者会让没对齐的模型对于一个问题生成多个(10个)回答，检查每个回答是否正确。给出一个信心值expected acc作为模型认知$k(x)$的一个近似

<img src=" ../../files/images/align-for-honesty/sample.png" style="zoom:33%;" >

### ABSOLUTE

设定一个阈值$\tau$，$k(x) = 1 \quad if \quad \text{expected acc} > \tau$。然后标数据的时候，把所有k(x)=-1的回答都改成了一个idk response

### CONFIDENCE

<figure class="half">
    <img src="../../files/images/align-for-honesty/numb.png" width="45%"/>
    <img src="../../files/images/align-for-honesty/verb.png" width="45%" />
</figure>

这里，作者标数据的时候直接把confidence写在回答里，然后按照正常SFT的办法

### MULTISAMPLE

刚才的absolute会根据一个阈值卡，这里作者直接把sample多次的每条数据当成单独的了，然后$k(x) = (c(x,y)==1)$。也就是说，标数据的时候，本来作对了的就不动，本来做错了的就改成一个idk response。

> 值得注意的是，这个方案会把训练集扩大M倍



## experiment

这里作者提了两个朴素的baseline：

- 原来的模型
- fine-tuned：在相同训练量上，使用turbo的answer进行SFT的模型
- prompt：上面提到的training-free方法
- 三种training方法，其中，$\tau$选取的是0.1

<figure class="half">
    <img src="../../files/images/align-for-honesty/main.png" width="40%"/>
    <img src="../../files/images/align-for-honesty/OOD.png" width="55%" />
</figure>

作者在TraivalQA数据集上做训练，使用Llama2-chat 7b作为基础模型，分别评测in-domain的traivalQA和OOD的另外三个数据集

效果如下：

- 发现基于训练的方法显著好于不训练的方法

- 相对来说，把confidence放到数据里，会让模型表现更好
- honesty属性在不同数据集上迁移能力较好，不管是ID还是OOD，加上confidence score都能让模型做的更好
- 直接finetune模型，会导致模型更加hallicinate，acc反而下降（这点在PKQA数据集表现得尤其明显）

接下来，作者探索了$\tau$对结果的影响，画了一张类似f1里面auc的图。发现，$\tau$越大，越容易把数据分类成模型不知道

- 因此idk数据越多，模型越容易变得over-confidence
- 另一方面，模型也越谨慎，所以prudence会提升，这里需要有一个权衡

<img src=" ../../files/images/align-for-honesty/auc.png" style="zoom:50%;" >

接下里，作者又做了scaling的实验：更大的模型会做得更好吗，更多的数据会做的更好吗？

<figure class="half">
    <img src="../../files/images/align-for-honesty/scale.png" width="35%"/>
    <img src="../../files/images/align-for-honesty/data-scale.png" width="60%" />
</figure>

首先，作者发现，confidence-based method对于所有模型规模效果都要更好一些

> 我发现：不同规模的模型对于Honesty的效果没啥区别，这说明了这个任务其实是挺困难的

其次，如果在训练集中加入MMLU的训练数据，对于Multi-sample方法的帮助很大，说明这个属性的习得也许是data-hungry的，模型需要更diverse的情况来判断自己的知识边界

> 不过，为啥Multisample+MMLU-data以后Acc下降这么多呢？



最后作者做了一些"对齐税"方面的实验，发现Honesty训练基本不会导致模型在别的任务表现下降。最后，作者总结了一下limitation和future，提了几个问题，我觉得还挺有意思的，分享给大家：

- 更好的k(x)：本篇工作用模型回答正确与否判断模型是否知道，这个在MMLU这种4选1中有误判假阳的情况
- confidence score能不能更好的利用？这里作者和calibration联系了一下
- 和RAG的结合：认知到自己知识边界的模型更清楚自己该怎么利用外界知识
- 和长文本的结合：需要结合reasoning的长文本场景的Honesty现在还没有研究，并且需要更细致的评测和训练



## 我的思考

很好的文章，formulation到method到实验设计都很顺滑，逻辑很完整，我看完了以后主要想到下面几个问题：

- 感觉可以评测一下turbo或者GPT4的表现？这里没做估计是因为需要一个unaligned模型去计算，没办法。要测也许只能给turbo来个un-align finetune，不知道是不是违规的
- scaling实验中，发现所有llama表现都差不多，说明这个能力也许是一个emergent的，甚至是reverse-scaling的？
- 这个能力，似乎是不能通过SFT习得的？因为每个模型都有自己的知识边界。作者也提到了，SFT-baseline will lead models to learn to hallicinate
