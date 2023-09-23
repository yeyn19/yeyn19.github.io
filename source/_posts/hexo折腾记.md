---
title: hexo折腾记
categories: 开发记录
tags:
  - 探索
  - 计算机
abbrlink: fa20dd9d
date: 2022-06-17 23:04:21
description: 用hexo建站都有哪些trick？
mathjax: true
---


今天卡发了半天hexo建站，还是遇到了不少bug，网上已有的教程大多是针对老版本的`hexo`和`Next`来说的，新版本的开发还是有一些新的trick。我在开发中大概遇到了这些问题：


#### 字数统计插件
使用了 `hexo-symbols-count-time`字数统计插件，其中中文占两个字符，每分钟阅读速度275。
发现所有的文章都显示`预计时间1min`，还以为是bug，找了好久最终原因竟然是：
- 所有$\leq 1 min$他都会显示约1min，是因为我前面的文章太短了…


#### 访问量统计
使用 `不蒜子` 的访问量统计工具，但是显示全是0。最后发现问题竟然是：
- 本地 `localhost` 不获取访问量，需要上 `deploy` 才能正确显示…
- 以及那个接口今年已经炸了，需要换个mirror: `//cdn.jsdelivr.net/npm/busuanzi@2.3.0`, 全文搜索 `busuanzi-count` 替换即可

#### 引用静态文件
发现使用 hash地址作为url以后所有的静态文件都不能引用了？
- 最后发现好像是路径地址计算的问题：所有的引用都要从`./source`作为地址的起点进行计算，同时`./_posts`文件夹似乎会被忽略？？最终解决办法是在`source`下新建了`./source/file`文件夹，把静态变量放到里面，再渲染


#### 相关文章推荐
这个是最蠢的…我用 `hexo-related-popular-posts` 库来实现相关文章推荐的功能，发现不管怎么样都推荐不了相关文章：
- 结果是他的推荐算法是根据tags的相关性来的，但是我那会只有一篇博客，所以算不出来相关文章，组建就被`disbale`了…

