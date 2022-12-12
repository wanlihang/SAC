import random
from copy import deepcopy

import numpy as np
import paddle
from paddle.distribution import Normal


class DE(object):
    def __init__(self, population, memory, factor=0.8, rounds=10, size=10, min_range=-1, max_range=1, cr=0.75):
        # 初始化种群，从 actor 池获取
        self.population = population

        print()
        # 初始化验证经验池
        self.memory = memory

        self.dimension = 2
        self.min_range = min_range
        self.max_range = max_range
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
        batch_state, batch_action, batch_reward, batch_next_state = self.memory
        episode_reward = float()
        for i in range(batch_state.shape[0]):
            state = batch_state[i, :]
            wait_time_list = list(state[-3:])
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

            reward_list = [1.0, 0.3, -0.7]
            if wait_time_list[action] == 0.0:
                episode_reward += reward_list[0]
            else:
                wait_time = wait_time_list[action]
                wait_time_list.sort()
                index = wait_time_list.index(wait_time)
                episode_reward += reward_list[index]

        return episode_reward

    def crossover_inplace(self, gene1, gene2):
        keys1 = list(gene1.state_dict())
        keys2 = list(gene2.state_dict())

        for key in keys1:
            if key not in keys2: continue

            # References to the variable tensors
            W1 = gene1.state_dict()[key]
            W2 = gene2.state_dict()[key]

            if len(W1.shape) == 2:  # Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                try:
                    num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
                except:
                    num_cross_overs = 1
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randint(0, W1.shape[0] - 1)  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = random.randint(0, W1.shape[0] - 1)  #
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1:  # Bias or LayerNorm
                if random.random() < 0.8: continue  # Crossover here with low frequency
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                # num_cross_overs = random.randint(0, int(num_variables * 0.05))  # Crossover number
                for i in range(1):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randint(0, W1.shape[0] - 1)  #
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = random.randint(0, W1.shape[0] - 1)  #
                        W2[ind_cr] = W1[ind_cr]

    # 变异
    def mutate(self):
        self.mutant = []
        for cur_i in range(self.size):
            select_range = [x for x in range(self.size)]
            select_range.remove(cur_i)
            r0, r1, r2 = np.random.choice(select_range, 3, replace=False)

            tmp = deepcopy(self.population[0])

            self.mutant.append(tmp)
            break

            for tmp_param, r0_param, r1_param, r2_param in zip(tmp.parameters(),
                                                               self.population[r0].parameters(),
                                                               self.population[r1].parameters(),
                                                               self.population[r2].parameters()):
                self.max_range = paddle.max(tmp_param)
                self.min_range = paddle.max(tmp_param)
                with paddle.no_grad():
                    if len(tmp_param.shape) == 1:
                        # 遍历 tensor，加入光荣的进化吧
                        for i in range(tmp_param.shape[0]):
                            value = r0_param[i] + (r1_param[i] - r2_param[i]) * self.factor
                            random_num = random.uniform(self.min_range, self.max_range)
                            if self.min_range <= value <= self.max_range:
                                tmp_param[i] = value
                            else:
                                tmp_param[i] = random_num

                    if len(tmp_param.shape) == 2:
                        # 遍历 tensor，加入光荣的进化吧
                        for i in range(tmp_param.shape[0]):
                            for j in range(tmp_param.shape[1]):
                                value = r0_param[i][j] + (r1_param[i][j] - r2_param[i][j]) * self.factor
                                random_num = random.uniform(self.min_range, self.max_range)
                                if self.min_range <= value <= self.max_range:
                                    tmp_param[i][j] = value
                                else:
                                    tmp_param[i][j] = random_num

            self.mutant.append(tmp)

    # 交叉选择
    def crossover_and_select(self):
        for cur_i in range(self.size):
            # 随机替换网络参数中数值
            for mutant_param, population_param in zip(self.mutant[cur_i].parameters(),
                                                      self.population[cur_i].parameters()):
                with paddle.no_grad():
                    if len(mutant_param.shape) == 1:
                        rand_i = random.randint(0, mutant_param.shape[0])
                        for i in range(mutant_param.shape[0]):
                            if random.random() > self.cr and i != rand_i:
                                mutant_param[i] = population_param[i]

                    if len(mutant_param.shape) == 2:
                        rand_i = random.randint(0, mutant_param.shape[0])
                        rand_j = random.randint(0, mutant_param.shape[1])
                        for i in range(mutant_param.shape[0]):
                            for j in range(mutant_param.shape[1]):
                                if random.random() > self.cr and i != rand_i and j != rand_j:
                                    mutant_param[i][j] = population_param[i][j]

                    reward = self.objective_function(self.mutant[cur_i])
                    # 超过原值则更新
                    if reward > self.object_function_values[cur_i]:
                        for target_param, param in zip(self.population[cur_i].parameters(),
                                                       self.mutant[cur_i].parameters()):
                            target_param.data.copy_(param.data)
                        self.object_function_values[cur_i] = reward

    def get_best(self):
        m = max(self.object_function_values)
        i = self.object_function_values.index(m)
        print("轮数：" + str(self.cur_round))
        print("最佳个体：" + str(self.population[i]))
        print("目标函数值：" + str(m))
        self.cur_round = self.cur_round + 1
        return self.population[i]

    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            best_policy = self.get_best()
        return best_policy
