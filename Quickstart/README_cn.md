
# 快速开始
ESBox提供两种训练方式：本地、分布式，可以根据以下指引来解决自定义问题。

## 1. 本地训练
**第一步：定义模型（可选）** 如果希望优化模型参数，请定义自己的模型网络，并继承模型基类（目前提供了`TorchModel`和`PaddleModel`），如果希望优化浮点型 List，请跳过这步。    
**第二步：定义问题** 需要继承`ProblemBase`基类并重写`evaluate`方法。（我们也提供了强化学习问题基类`RLProblem`和数学函数基类`FuncProblem`，使用方法详见[Examples](http://gitlab.baidu.com/nlp-ol/ESBox/tree/developing/examples)）    
**第三步：** 在配置文件`config.yaml`中定义算法参数以及任务相关的参数，并使用`esbox.core.Task`构建训练任务，详见`run_local.py`。

## 2. 分布式训练 
与上述**本地训练**基本一致，不同的地方有三点
+ 在**第二步**中，需要额外使用`@parl.remote_class(wait=False)`装饰自己的问题类。
+ 在**第三步**中，需要在配置文件中配置`xparl_addr`和`num_workers`。
+ 在**第三步**中，使用`esbox.core.ParallelTask`构建训练任务，详见`run_distributed.py`。

## 使用样例
以下是使用 OpenAI-ES 算法优化 List 和 Model 的两个使用样例，你可以根据指引修改它们以构建自己的问题。每个样例都提供了本地训练和分布式训练两种训练方式。
#### 1. 强化学习问题 CartPole-v1
该样例基于 paddlepaddle 深度学习框架，使用 OpenAI-ES 算法优化**模型网络参数**解决强化学习中的 CartPole-v1 问题。 

```bash
cd CartPole-example
```
- 启动本地训练
    ```bash
    python run_local.py --config_file ./local.ymal
    ```
- 启动分布式训练
    ```bash
    xparl start --port 8010
    python run_distributed.py --config_file ./distributed.ymal
    ```
**预期结果** 
200 轮优化迭代后，该任务可在环境中获取 500 分左右的奖励。

#### 2. 二维二次函数问题
该样例使用 OpenAI-ES 算法优化**浮点型列表**（二维二次函数的输入）求取一个二维二次函数的最大值。

```bash
cd Quadratic-example
```
- 启动本地训练
    ```bash
    python run_local.py --config_file ./local.ymal
    ```
- 启动分布式训练
    ```bash
    xparl start --port 8010
    python run_distributed.py --config_file ./distributed.ymal
    ```
**预期结果** 
200 轮优化迭代后，该任务可找到函数最大值 0.99999（实际最大值1）。
