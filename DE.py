import random
from copy import deepcopy

import numpy as np
import paddle
from paddle.distribution import Normal


class DE(object):
    def __init__(self, population, batch_state, factor=0.8, rounds=10, size=10, cr=0.8):
        # 初始化种群，从 actor 池获取
        self.population = population

        # 初始化验证经验池
        self.batch_state = batch_state

        self.dimension = 2
        self.factor = factor
        self.rounds = rounds
        self.size = size
        self.cur_round = 1
        self.cr = cr
        self.object_function_values = [self.objective_function(v) for v in self.population]
        self.mutant = None

    # 目标函数
    def objective_function(self, actor):
        # 计算当前 actor 所有累计 episode_reward
        episode_reward = float()
        for i in range(self.batch_state.shape[0]):
            state = self.batch_state[i, :]
            wait_time_list = list(state[-3:])
            state = np.array(state)
            state = paddle.to_tensor(state.reshape(1, -1), dtype='float32')
            act_mean, act_log_std = actor(state)
            normal = Normal(act_mean, act_log_std.exp())
            # 重参数化  (mean + std*N(0,1))
            x_t = normal.sample([1])
            action = paddle.tanh(x_t)

            log_prob = normal.log_prob(x_t)
            log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
            log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)

            action, _ = action[0], log_prob[0]

            action_numpy = action.cpu().numpy()[0]

            action = action_numpy

            # 动作转换
            if action <= - 1.0 / 3:
                action = 0
            elif action <= 1.0 / 3:
                action = 1
            else:
                action = 2

            episode_reward -= wait_time_list[action] / 1000

            # reward_list = [1.0, 0.3, -0.7]
            # if wait_time_list[action] == 0.0:
            #     episode_reward += reward_list[0]
            # else:
            #     wait_time = wait_time_list[action]
            #     wait_time_list.sort()
            #     index = wait_time_list.index(wait_time)
            #     episode_reward += reward_list[index]

        return episode_reward

    # 变异
    def mutate(self):
        self.mutant = []
        for cur_i in range(self.size):
            select_range = [x for x in range(self.size)]
            select_range.remove(cur_i)
            r0, r1, r2 = random.sample(select_range, 3)

            tmp = deepcopy(self.population[0])

            for tmp_param, r0_param, r1_param, r2_param in zip(tmp.parameters(),
                                                               self.population[r0].parameters(),
                                                               self.population[r1].parameters(),
                                                               self.population[r2].parameters()):
                with paddle.no_grad():
                    tmp_param.set_value(r0_param + (r1_param - r2_param) * self.factor)

            self.mutant.append(tmp)

    # 交叉选择
    def crossover_and_select(self):
        for cur_i in range(self.size):
            # 随机替换网络参数中数值
            for mutant_param, population_param in zip(self.mutant[cur_i].parameters(),
                                                      self.population[cur_i].parameters()):
                with paddle.no_grad():
                    if len(mutant_param.shape) == 1:
                        if random.random() < self.cr:
                            continue
                        for i in range(1):
                            receiver_choice = random.random()
                            if receiver_choice < 0.5:
                                ind_cr = random.randint(0, mutant_param.shape[0] - 1)  #
                                mutant_param[ind_cr] = population_param[ind_cr]
                            else:
                                ind_cr = random.randint(0, mutant_param.shape[0] - 1)  #
                                population_param[ind_cr] = mutant_param[ind_cr]
                        self.evolution_network(cur_i)
                    if len(mutant_param.shape) == 2:
                        num_variables = mutant_param.shape[0]
                        num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
                        for i in range(num_cross_overs):
                            receiver_choice = random.random()  # Choose which gene to receive the perturbation
                            if receiver_choice < 0.5:
                                ind_cr = random.randint(0, mutant_param.shape[0] - 1)  #
                                mutant_param[ind_cr, :] = population_param[ind_cr, :]
                            else:
                                ind_cr = random.randint(0, mutant_param.shape[0] - 1)  #
                                population_param[ind_cr, :] = mutant_param[ind_cr, :]

    def evolution_network(self, cur_i):
        reward = self.objective_function(self.mutant[cur_i])
        old_reward = self.object_function_values[cur_i]
        # 超过原值则更新
        if reward > old_reward:
            for target_param, param in zip(self.population[cur_i].parameters(),
                                           self.mutant[cur_i].parameters()):
                target_param.set_value(param)
            self.object_function_values[cur_i] = reward

    def get_best(self):
        m = max(self.object_function_values)
        i = self.object_function_values.index(m)
        # print("轮数：" + str(self.cur_round))
        # print("最佳个体：" + str(self.population[i]))
        # print("目标函数值：" + str(m))
        self.cur_round = self.cur_round + 1
        return self.population[i]

    def evolution(self):
        best_policy = self.get_best()
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            best_policy = self.get_best()
        return best_policy
