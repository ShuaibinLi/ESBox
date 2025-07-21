
[English](./README.md) | 简体中文

> ESBox是一个高效的黑盒优化工具，具有多种进化策略算法。


## ESBox 功能一览
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>内置问题</b>
      </td>
      <td>
        <b>算法</b>
      </td>
      <td>
        <b>用户接口</b>
      </td>
    </tr>
    <tr valign="top">
      <td align="left" >
      <ul><li><b>数学函数</b></li>
        <ul>
          <li>Ackley</li>
          <li>Griewank</li>
          <li>Rastrigin</li>
          <li>Rosenbrock</li>
          <li>StyblinskiTang</li>
          <li>Zakharov</li>
        </ul>
        </ul>
      <ul>
        <li><b>强化学习环境 gym</b></li>
          <ul>
           <li><a href="https://mujoco.org/">Mujoco</a></li>
                <ul><li>HalfCheetah-v5</li></ul>
                <ul><li>Humanoid-v5</li></ul>
          </ul>
      </ul>
      </td>
      <td align="left" >
        <ul>
        <li><b>OpenAI-ES</b><a href="https://arxiv.org/abs/1803.07055"> 论文</a></li>
            <ul>
            <li>高斯 采样器（对称采样）</li>
            <li>OpenAIES 学习器</li>
            </ul>
        <li><b>ARS</b><a href="https://arxiv.org/abs/1803.07055"> 论文</a></li>
            <ul>
            <li>高斯 采样器（对称采样）</li>
            <li>ARS 学习器</li>
            </ul>
        <li><b>NSRAES</b><a href="https://arxiv.org/abs/1703.03864"> 论文</a></li>
            <ul>
            <li>高斯 采样器（对称采样）</li>
            <li>NSRAES 学习器</li>
            </ul>
        <li><b>CMA-ES</b><a href="https://arxiv.org/abs/1604.00772"> 论文</a></li>
            <ul>
            <li>CMA 采样器</li>
            <li>CMAES 学习器</li>
            </ul>
        <li><b>Sep-CMA-ES</b><a href="https://hal.inria.fr/inria-00270901v4"> 论文</a></li>
            <ul>
            <li>Sep-CMA 采样器</li>
            <li>Sep-CMAES 学习器</li>
            </ul>
        </ul>
      </td>
      <td align="left" >
        <li><a href="examples/tuned_configs/">配置文件</a></li>
        <li><b>优化目标</b></li>
            <ul>
            <li>Model（torch，paddlepaddle）</li>
            <li>List（浮点型）</li>
            </ul>
        <li><b><a href="examples/">例子</a></b></li>
            <ul>
            <li>本地训练</li>
              <ul> 
              <li>数学函数（List，Model） </li>
              </ul>
            <li>分布式训练</li>
              <ul> 
              <li>强化学习问题 HalfCheetah-v5（Model）</li>
              </ul>
            </ul>
          <li><b><a href="Quickstart/">使用样例</a></b></li>
            <ul>
            <li>强化学习问题：Cartpole（优化 Model）</li>
              <ul> 
              <li>本地训练</li>
              <li>分布式训练</li>
              </ul>
            <li>函数问题：2维2次函数（优化浮点型 List）</li>
              <ul> 
              <li>本地训练</li>
              <li>分布式训练</li>
              </ul>
            </ul>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## 如何使用

### 安装
```
git clone https://github.com/ShuaibinLi/ESBox.git
cd ESBox
pip install . 
```
### 其他依赖
+ [parl](https://github.com/PaddlePaddle/PARL)
+ pytorch or paddlepaddle
+ [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
注意：使用 mujoco或atari 环境，使用`pip install "gymnasium[all]"`安装 gymnasium.


### 使用教程
#### [快速开始](Quickstart/)
+ 两种训练方式：本地训练、分布式训练
+ 两种优化形式：模型优化（以 CartPole-v1 为例）、浮点型 List 优化（以2维2次函数为例）

#### [Examples](examples/)
+ 提供五种算法解决两大类问题的示例
+ 展示 benchmark 复现结果
