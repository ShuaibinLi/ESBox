
## Examples
Examples and results of five algorithms on the two built-in problems.
### Benchmark result
+ Results of four algorithms in five dimensions (2, 10, 20, 100, 500) on two mathematical functions (Ackley and Griewank). (Direct optimization of float list, i.e. inputs to mathematical functions)
    <p align="center">
    <img src=".results/ackley_results.png" alt="ackley_results">
    </p>
    <p align="center">
    <img src=".results/griewank_results.png" alt="griewank_results">
    </p>       
    Note: 1. NSRA-ES is of little significance in this question, so no comparative experiments are conducted. 2. At dim=500, the difficulty of the problem increases, and we increase the sample_num to achieve the convergence effect (sample_num=200 for ARS, sample_num=500 for CMAES and Sep-CMAES).  

+ Results of five algorithms in two reinforcement learning environments (HalfCheetah-v2 and Humanoid-v2). (model network for optimization)
    <center class="half">
        <img src=".results/HalfCheetah-v2.png" width="350" alt="HalfCheetah-v2"/><img src=".results/Humanoid-v2.png" width="350" alt="Humanoid-v2"/>
    </center>   
    Note: In the Humanoid-v2 environment, there are two algorithms that do not do comparative experiments, because the observation space and action space dimensions are too large, the NSRAES algorithm is difficult to calculate bc, and the CMAES sampling algorithm is too expensive (it takes 1-2 minutes to sample once).

## How to use
### Local training for function problems
First, go to the [examples] (http://gitlab.baidu.com/nlp-ol/ESBox/tree/developing/examples) folder and reproduce the above experimental results according to the following guidelines.
```bash
cd examples
```

### Function problems (local training)
For function problems, this example provides two different solutions (Take using the CMA-ES algorithm to solve a two-dimensional ackley function as an example).
- Way 1. Directly optimize the input of the function, start local training
    ```bash
    python run_function.py --config_file ./tuned_configs/cmaes_function.ymal
    ```
- Way 2. By optimizing a single-layer model network with a fixed input of 1 (the output of the model is the input of the function)
    ```bash
    python run_function_model.py --config_file ./tuned_configs/cmaes_function_model.ymal
    ```

### RL problems (distributed training)
For RL environments, we provide examples of direct optimization policy models (Take using OpenAI-ES algorithms to solve the HalfCheetah-v2 problem as an example).  
Before starting training, please use [xparl](https://parl.readthedocs.io/en/latest/parallel_training/setup.html) to create a cluster. Refer to the [documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html) for more information on clustering..

```bash
# create a cluster
xparl start --port 8010 --cpu_num 50

# start training
python run_mujoco.py --config_file ./tuned_configs/openaies_mujoco.ymal
```
For configuration of the remaining algorithms and problems
, see [tuned_config](http://gitlab.baidu.com/nlp-ol/ESBox/tree/developing/examples/tuned_configs).
