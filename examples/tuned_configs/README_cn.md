
# 配置文件
一个配置文件`config.ymal`主要包含三部分相关的内容：**算法**，**任务** 和 **问题**（可选，可修改内置问题的配置如`RLProblem`和`FuncProblem`）。      
每一部分的详细配置说明如下。

## 算法相关
### 采样器配置
+ **`sample_num (int)`**: 一次迭代优化过程中的采样数量
+ **`mirror_sample (bool, optinal, default=true)`**: 是否使用对称采样的技巧（针对高斯采样器）
+ **`noise_stdev (float)`**: 噪声标准差
+ **`seed (int, optinal, default=123)`**: 采样器的随机种子
+ **`init_policy (string, optinal, default='randome')`**: 被优化参数的初始化策略（可选项：'zeros', 'ones', 'random', 'uniform', 'normal'）

### 学习器配置
+ **`learning_rate (float)`**: 学习率
+ **`top_k (int, optinal, default=None)`**: 表现最优的前k个评估点（ARS、CMAES、Sep-CMAES），计算新颖性时使用的最近邻数（NSRA-ES）
+ **`init_sigma (float, optinal, default=1.0)`**: 协方差矩阵初始值（CMAES、Sep-CMAES）
+ **`alg_name (string, )`**: 算法名称（可选项： 'openaies', 'ars', 'nsraes', 'cmaes', 'sep-cmaes'）
**提示**：不同学习器的具体参数说明可查看[algorithms](http://gitlab.baidu.com/nlp-ol/ESBox/tree/developing/esbox/algorithms)中相对应的注释。

## 任务配置
+ **`max_runs (int, optinal, default=200)`**: 最大迭代优化次数
+ **`display (bool, optinal, default=True)`**: 是否打印训练过程中的评估信息
+ **`eval_every_run (int, optinal, default=10)`**: 评估训练的间隔轮次
+ **`xparl_addr (string, optinal, default=None)`**: 分布式训练的 xparl 地址，比如：localhost:8010（仅分布式训练方式需要）
+ **`num_workers (int, optional)`**: （`xparl_addr` 的配套参数）使用的分布式训练的处理器个数（仅分布式训练方式需要）


## 问题配置 (可选项，适用于内置问题类)
+ ### 数学函数问题
    + **`func_name` (string)**: 内置数学函数的名字，比如：ackley, griewank,   zakharov, rastrigin, rosenbrock, styblinskitang.
    + **`dim` (int)**: 数学函数的维度，[1, $+\infty$]，通常是 [1, 100]
    + **`bounds` (list, optional, default=None)**: 输入的限制，比如：[-5, 5]

    + **`scale` (bool, optional, default=False)**: 是否使用防缩映射将数学函数的输入映射到上下界范围内（通常用在最后使用 tanh 激活函数的模型网络解决数学函数问题中）

+ ### 强化学习环境
    + **`env_name (string)`**: 环境名称，比如 HalfCheetah-v2
    + **`seed (int, optinal)`**: 环境种子（同采样器的种子，一个配置文件仅需要出现一次即可）