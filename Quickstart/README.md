
# QuickStart
ESBox provides two training methods: local and distributed, and you can solve custom problems according to the following guidelines.

## 1. Local training 
**Step 1. Define model (Optional)** If you want to optimize model parameters, define your own model network and inherit the base model class (`TorchModel` and `PaddleModel` are currently available). If you want to optimize your float List, skip this step.         
**Step 2. Define problem** You need to inherit the `ProblemBase` for your problem and override the `evaluate` method. (We also provide other two base problem classes: `RLProblem` and `FuncProblem`, the usage of which can be found in [Examples](../examples))          
**Step 3.** Define algorithm hyperparameter and task-related parameters in the configuration file `config.yaml`, and use `esbox.core.Task` to build a training task, see `run_local.py`.

## 2. Distributed training 
It is basically the same as the **local training** above, three differences
+ In **Step 2**, additionally decorate your problem class with `@ray.remote`.
+ In **Step 3**, `ray_addr` and `num_workers` are necessary in the configuration file.
+ In **Step 3**, use `esbox.core.ParallelTask` to build a training task, see `run_distributed.py`.

## Examples
Here are two examples of using OpenAI-ES algorithms to optimize List and Model, and you can modify them according to the guidelines to build your own questions. Each example provides two training methods: local training and distributed training.

#### 1. RL problem: CartPole-v1
This example is based on the paddlepaddle deep learning framework and uses the OpenAI-ES algorithm to optimize **model** network parameters to solve the CartPole-v1 problem.

```bash
cd CartPole-example
```
- start local training
    ```bash
    python run_local.py --config_file ./local.ymal
    ```
- start distributed training
    ```bash
    ray start --head --num-cpus 50 --port 6379
    python run_distributed.py --config_file ./distributed.ymal
    ```
**Expected Result** 
The task can get around 500 points after 200 steps.

#### 2. A 2-dimentional quadratic function problem
This example uses the OpenAI-ES algorithm to optimize float **list** (as input of 2D quadratic function) to find the maximum value of a 2-dimension quadratic function.

```bash
cd Quadratic-example
```
- start local training
    ```bash
    python run_local.py --config_file ./local.ymal
    ```
- start distributed training
    ```bash
    ray start --head --num-cpus 50 --port 6379
    python run_distributed.py --config_file ./distributed.ymal
    ```

**Expected Result** 
The task can get around 0.99999 ( the actual maximum value is 1.0 ) points after 200 steps.
