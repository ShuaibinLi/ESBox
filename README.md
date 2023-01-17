
English | [简体中文](./README_cn.md)

> ESBox is an efficient tool for black-box optimization with multiple evolutionary strategy algorithms.


## ESBox Capabilities in a Glance
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Build-in Problems</b>
      </td>
      <td>
        <b>Algorithms</b>
      </td>
      <td>
        <b>User APIS</b>
      </td>
    </tr>
    <tr valign="top">
      <td align="left" >
      <ul><li><b>Function problems</b></li>
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
        <li><b>RL problems</b></li>
          <ul>
           <li><a href="https://mujoco.org/">Mujoco</a></li>
                <ul><li>HalfCheetah-v2</li></ul>
                <ul><li>Humanoid-v2</li></ul>
          </ul>
      </ul>
      </td>
      <td align="left" >
        <ul>
        <li><b>OpenAI-ES</b><a href="https://arxiv.org/abs/1803.07055"> paper</a></li>
            <ul>
            <li>Gaussian sampler (Mirror)</li>
            <li>OpenAIES learner</li>
            </ul>
        <li><b>ARS</b><a href="https://arxiv.org/abs/1803.07055"> paper</a></li>
            <ul>
            <li>Gaussian sampler (Mirror)</li>
            <li>ARS learner</li>
            </ul>
        <li><b>NSRAES</b><a href="https://arxiv.org/abs/1703.03864"> paper</a></li>
            <ul>
            <li>Gaussian sampler (Mirror)</li>
            <li>NSRAES learner</li>
            </ul>
        <li><b>CMA-ES</b><a href="https://arxiv.org/abs/1604.00772"> paper</a></li>
            <ul>
            <li>CMA sampler</li>
            <li>CMAES learner</li>
            </ul>
        <li><b>Sep-CMA-ES</b><a href="https://hal.inria.fr/inria-00270901v4"> paper</a></li>
            <ul>
            <li>Sep-CMA sampler</li>
            <li>Sep-CMAES learner</li>
            </ul>
        </ul>
      </td>
      <td align="left" >
        <li><a href="examples/tuned_configs/">Config</a></li>
        <li><b>Objects</b></li>
            <ul>
            <li>Model (torch, paddlepaddle)</li>
            <li>List (float)</li>
            </ul>
        <li><b><a href="examples/">Examples</a></b></li>
            <ul>
            <li>Local training</li>
              <ul> 
              <li>Function (List, Model) </li>
              </ul>
            <li>Distributed training</li>
              <ul> 
              <li>RL problem: HalfCheetah-v2 (Model) </li>
              </ul>
            </ul>
          <li><b><a href="Quickstart/">QuickStart</a></b></li>
            <ul>
            <li>RL problem: Cartpole-v1 (optimization Model) </li>
              <ul> 
              <li>local training </li>
              <li>distributed training </li>
              </ul>
            <li>Function problem: Quadratic function (optimization float List) </li>
              <ul> 
              <li>local training </li>
              <li>distributed training </li>
              </ul>
            </ul>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## How to use

### Install
```
git clone https://github.com/ShuaibinLi/ESBox.git
cd ESBox
pip install . 
```
### Other Dependencies
+ [parl](https://github.com/PaddlePaddle/PARL)
+ pytorch or paddlepaddle
+ gym==0.18.0
+ mujoco-py==2.1.2.14   
Note: To use mujoco-v2 env in gym0.18.0, please download the [mujoco210](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) binaries for Linux and extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.


### Start training
#### [QuickStart](Quickstart/)
+ Two training methods: local training and distributed training
+ Two forms of optimization: Model optimization (CartPole-v1 as an example) and float List optimization (2-dimensional 2nd degree function as an example)

#### [Examples](examples/)
+ Examples of five algorithms that solve two categories of problems
+ The results of the benchmark reproduction