import os
import time
import numpy as np
from loguru import logger
from tensorboardX import SummaryWriter

from esbox.sampler import *
from esbox.algorithms import *

__all__ = ['Task']


class Task(object):
    """ Task for problems
    Args:
        config (dict): Config of the problem and algorithm
        eval_func (class): user defined evaluation function
    """

    def __init__(self, config, eval_func):
        self.config = config

        # training configs
        self.max_runs = self.config.hyparams.get('max_runs', 1000)
        self.eval_every_run = self.config.hyparams.get('eval_every_run', 10)
        self.display = self.config.hyparams.get('display', True)
        self.work_dir = self.config.hyparams.get('work_dir')

        ### parameters for problem
        self.func_name = self.config.hyparams.get('func_name', None)
        self.env_name = self.config.hyparams.get('env_name', None)
        self.dim = self.config.hyparams.get('dim', None)
        self.scale = self.config.hyparams.get('scale', False)
        self.bounds = self.config.hyparams.get('bounds', None)
        if self.config.model_cls:
            self.bounds = None
            self.eval_episodes = 3
        else:
            self.eval_episodes = 1

        ### parameters for sampler
        self.mirror = self.config.hyparams.get('mirror_sample', None)
        self.seed = self.config.hyparams.get('seed', None)
        # gaussian
        self.noise_stdev = self.config.hyparams.get('noise_stdev', None)

        ### parameters for learner
        self.alg_name = self.config.alg_name
        self.learning_rate = self.config.hyparams.get('learning_rate', None)
        self.sample_num = self.config.hyparams.get('sample_num', None)
        # openaies
        self.l2_coeff = self.config.hyparams.get('l2_coeff', None)
        # ars
        self.top_k = self.config.hyparams.get('top_k', None)
        self.shift = self.config.hyparams.get('shift', None)
        # nsraes
        self.n = self.config.hyparams.get('n', None)
        self.meta_population_size = self.config.hyparams.get('meta_population_size', None)
        # cmaes
        self.init_sigma = self.config.hyparams.get('init_sigma', None)
        # self.pop_size = self.config.hyparams.get('pop_size', None)
        self.mu = self.config.hyparams.get('mu', None)

        # initialization of problem
        if self.func_name is not None:
            self.eval_func = eval_func(self.func_name, dim=self.dim, scale=self.scale)
        elif self.env_name is not None:
            self.eval_func = eval_func(self.env_name, self.seed, n=self.n)
        else:
            self.eval_func = eval_func()
        # get problem dim
        self.param_num = self.eval_func.get_dim()
        self.config.hyparams['param_num'] = self.param_num
        self.timesteps = 0

        # init weights
        init_policy = self.config.hyparams.get('init_policy', 'random')
        self.init_weights = self._init_weights(init_policy)

        # print configs and init_weights
        logger.add(os.path.join(self.work_dir, 'task_run.log'))
        logger.info("Running task with config: \n{}, \nInit weights are: \n{}".format(
            (self.config.alg_name, self.config.hyparams), self.init_weights))

        ## initialization of sampler and learner
        if self.alg_name == 'openaies':
            self.sampler = GaussianSampler(self.param_num,
                                           self.noise_stdev,
                                           self.seed,
                                           mirro_sampling=self.mirror,
                                           bounds=self.bounds)
            self.learner = OpenAIES(self.param_num, self.learning_rate, self.l2_coeff, init_weights=self.init_weights)
        elif self.alg_name == 'ars':
            self.sampler = GaussianSampler(self.param_num,
                                           self.noise_stdev,
                                           self.seed,
                                           mirro_sampling=self.mirror,
                                           bounds=self.bounds)
            self.learner = ARS(self.param_num, self.learning_rate, self.top_k, init_weights=self.init_weights)
        elif self.alg_name == 'nsraes':
            self.sampler = GaussianSampler(self.param_num, self.noise_stdev, self.seed, mirro_sampling=self.mirror)
            self.learner = NSRAES(weights_size=self.param_num,
                                  step_size=self.learning_rate,
                                  k=self.top_k,
                                  pop_size=self.sample_num,
                                  sigma=self.noise_stdev,
                                  init_weights=self.init_weights)
            print("Collecting init archives for narses ...")
            while len(self.learner._archives) < self.meta_population_size:
                rewards, eval_info = self.evaluate()
                max_idx = np.argsort(rewards)[::-1][:self.meta_population_size]
                for idx in max_idx:
                    self.learner._archives.append(eval_info['bcs'][idx])
                    self.learner.latest_r = np.mean(rewards)
        elif self.alg_name == 'cmaes':
            self.sampler = CMASampler(weights_size=self.param_num,
                                      bounds=self.bounds,
                                      seed=self.seed,
                                      sigma=self.init_sigma)
            self.learner = CMAES(weights_size=self.param_num,
                                 sigma=self.init_sigma,
                                 population_size=self.sample_num,
                                 mu=self.mu,
                                 cov=None,
                                 init_weights=self.init_weights)
        elif self.alg_name == 'sep-cmaes':
            self.sampler = SepCMASampler(weights_size=self.param_num,
                                         bounds=self.bounds,
                                         seed=self.seed,
                                         sigma=self.init_sigma)
            self.learner = SepCMAES(weights_size=self.param_num,
                                    sigma=self.init_sigma,
                                    population_size=self.sample_num,
                                    mu=self.mu,
                                    cov=None,
                                    init_weights=self.init_weights)
        else:
            raise NotImplementedError("ESbox hasnot implemented {} algorithm.".format(self.alg_name))
        self.learned_info = {}

    def _init_weights(self, init_policy):
        """init policy: 
            supported initialization methods: zeros, ones, uniform, normal, random,
            TODO
            xavier_uniform, xavier_normal, 
            kaiming_uniform, kaiming_normal,
        """
        np.random.seed(self.seed)
        if init_policy == 'zeros':
            init_weights = np.zeros(self.param_num)
        elif init_policy == 'ones':
            init_weights = np.ones(self.param_num)
        elif init_policy == 'random':
            init_weights = np.random.random(self.param_num)
        elif init_policy == 'uniform':
            init_weights = np.random.uniform(-1, 1, self.param_num)
        elif init_policy == 'normal':
            init_weights = np.random.normal(0, 1, self.param_num)
        else:
            raise KeyError("Unsupported init policy %s" % init_policy)
        return init_weights

    def run_evals(self):
        """ Aggregate update step from rollouts.

        Returns:
            rollout_rewards (numpy.array): array of rewards for rollouts.
                            shape (sample_num, 1, param_num) for mirror_sampling=False
                                  (sample_num // 2, 2, param_num) for mirror_sampling=True
            sampled_info (dict): different samplers return different dict, all of which contain 'batch_flatten_weights' key.
        """
        self.latest_weights = self.learner.weights
        if self.alg_name == 'nsraes':
            rewards, eval_info = self.evaluate()
            max_idx = np.argsort(rewards)[::-1]
            self.learner._archives.append(eval_info['bcs'][max_idx[0]])
            self.learner.latest_r = np.mean(rewards)

        # sample n batch weights
        sample_batch = self.sample_num // 2 if self.mirror else self.sample_num
        sampled_info = self.sampler(self.latest_weights, sample_batch, self.learned_info)

        batch_flatten_weights = sampled_info['batch_flatten_weights']

        # evaluates sampled weights
        sampled_results = self.eval_func.evaluate_batch(batch_flatten_weights)
        noise_rewards = np.array(sampled_results['values'])
        sampled_info['bcs'] = sampled_results['info'].get('bcs', None)
        return noise_rewards, sampled_info

    def evaluate(self):
        """Runs policy evaluation with the latest weights.

        Returns:
            rewards (list): len == num_workers, rewards collected during the evaluation
            eval_info (dict): additional information during the evaluation, e.g. steps, bcs(for nsraes alg)
        """
        weights = self.learner.weights
        eval_values = []
        bcs = []
        steps = []
        for _ in range(self.eval_episodes):
            result = self.eval_func.evaluate(weights)
            eval_values.append(result['value'])
            bcs.append(result['info'].get('bc', weights))
            steps.append(result['info'].get('step', 1))
        eval_values = np.array(eval_values).squeeze()
        eval_info = {"steps": np.array(steps), 'bcs': bcs}
        return eval_values, eval_info

    def run(self):
        """Trains and evaluate the model.
        """
        self.writer = SummaryWriter(log_dir=os.path.join(self.work_dir, "tb_res"))
        for i in range(self.max_runs):
            # Perform one update step of the policy weights.
            rollout_rewards, sampled_info = self.run_evals()
            self.learned_info = self.learner.learn(rollout_rewards, sampled_info)
            if self.display:
                logger.info("Training step: {}, avg reward: {}, max reward: {}".format(
                    i + 1, np.mean(rollout_rewards), np.max(rollout_rewards)))
            self.writer.add_scalar('train/avg_reward', np.mean(rollout_rewards), i + 1)
            self.writer.add_scalar('train/std_reward', np.std(rollout_rewards), i + 1)
            self.writer.add_scalar('train/max_reward', np.max(rollout_rewards), i + 1)
            self.writer.add_scalar('train/min_reward', np.min(rollout_rewards), i + 1)
            self.writer.add_scalar('train/total_steps', self.timesteps, i + 1)

            # record statistics every `eval_every_run` iterations
            if (i == 0 or (i + 1) % self.eval_every_run == 0):
                rewards, _ = self.evaluate()

                if self.eval_episodes == 1:
                    logger.info("Steps: {}, eval_reward: {}".format(i + 1, np.mean(rewards)))
                    # self.writer.add_scalar('eval/reward', np.mean(rewards), i + 1)
                    # self.writer.add_scalar('eval/std_reward', np.std(rewards), i + 1)
                else:
                    logger.info("Steps: {}, Avg_eval_reward: {}, Max_eval_reward: {}".format(
                        i + 1, np.mean(rewards), np.max(rewards)))
                    self.writer.add_scalar('eval/avg_reward', np.mean(rewards), i + 1)
                    self.writer.add_scalar('eval/std_reward', np.std(rewards), i + 1)
                    self.writer.add_scalar('eval/max_reward', np.max(rewards), i + 1)
                    self.writer.add_scalar('eval/min_reward', np.min(rewards), i + 1)
        self.writer.close()
