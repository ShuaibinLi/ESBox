
# Config  
A `config.ymal` file mainly includes the relevant configuration of the three parts: **algorithm**, **task** and **problem** (optional, you can modify the configuration of built-in issues such as `RLProblem` and `FuncProblem`).
The detailed configuration of each part is described below.


## Algorithm settings
### sampler parameters
+ **`sample_num (int)`**: number of samples in an iterative optimization process
+ **`mirror_sample (bool, optinal, default=true)`**: whether to use mirror sampling techniques (for Gaussian samplers)
+ **`noise_stdev (float)`**: standard deviation of noise
+ **`seed (int, optinal, default=123)`**: random seed for sampler
+ **`init_policy (string, optinal, default='randome')`**: initialization strategy of optimized parameters (options: 'zeros', 'ones', 'random', 'uniform', 'normal')

### learner parameters
+ **`learning_rate (float)`**: learning rate for optimizer
+ **`top_k (int, optinal, default=None)`**: top k sample weights with optimal performance (for ARS, CMAES, Sep-CMAES), or number of nearest neighbors used when calculating novelty (for NARA-ES)
+ **`init_sigma (float, optinal, default=1.0)`**: initial standard deviation of covariance matrix (for CMAES, Sep-CMAES)
+ **`alg_name (string, )`**: name of algorithm (options: 'openaies', 'ars', 'nsraes', 'cmaes', 'sep-cmaes')     
**Note**: The specific parameter descriptions of different learners can be found in the corresponding comments in [algorithms] (http://gitlab.baidu.com/nlp-ol/ESBox/tree/developing/esbox/algorithms).

## Task settings
+ **`max_runs (int, optinal, default=200)`**: Max time steps to run environment
+ **`display (bool, optinal, default=True)`**: whether to display the training process
+ **`eval_every_run (int, optinal, default=10)`**: step interval between two consecutive evaluations
+ **`xparl_addr (string, optinal, default=None)`**: xparl address for distributed training, e.g. localhost:8010 (Required only for distributed training)
+ **`num_workers (int, optional)`**: (Supporting parameter of `xparl_addr`) number of workers used to evaluation problems (Required only for distributed training)


## Problem settings (Optional, required when using built-in problems)
+ ### function problems
    + **`func_name` (string)**: function name of build-in functions problems, e.g. ackley, griewank, zakharov, rastrigin, rosenbrock, styblinskitang.
    + **`dim` (int)**: dimension of the problem, [1, $+\infty$], usually [1, 100]
    + **`bounds` (list, optional, default=None)**: constraint bounds on the input of function, e.g. [-5, 5]

    + **`scale` (bool, optional, default=False)**: whether to implement linear scaling from bounds to [x_low, x_up], default [-1, 1]. Usually used when using networks, and the last layer of the network is tanh.

+ ### gym problems
    + **`env_name (string)`**: name of gym environment, e.g. HalfCheetah-v2
    + **`seed (int, optinal)`**: seed for env (with the seed of the sampler, a config file only needs to appear once)
