import math
from copy import deepcopy

import numpy as np
import random

import paddle


class DE(object):
    def __init__(self, population, memory, factor=0.8, rounds=100, size=10, min_range=-1, max_range=1, cr=0.75):
        # 初始化种群，从 actor 池获取
        self.individuality = population

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
        self.object_function_values = [self.objective_function(v) for v in self.individuality]
        self.mutant = None

    # 目标函数
    def objective_function(self, actor):
        # 计算当前 actor 所有累计 reward

        return

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

    def mutate_inplace(self, gene):
        mut_strength = 0.1
        num_mutation_frac = 0.05
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.02

        num_params = len(list(gene.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

        for i, param in enumerate(gene.parameters()):  # Mutate each param

            # References to the variable keys
            W = param.data
            if len(W.shape) == 2:  # Weights, no bias

                num_weights = W.shape[0] * W.shape[1]
                ssne_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_mutations = random.randint(0, int(math.ceil(
                        num_mutation_frac * num_weights)))  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim1 = random.randint(0, W.shape[0] - 1)
                        ind_dim2 = random.randint(0, W.shape[-1] - 1)
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                        elif random_num < reset_prob:  # Reset probability
                            W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
                        else:  # mutauion even normal
                            W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

                        # Regularization hard limit
                        # W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], self.args.weight_magnitude_limit)

            elif len(W.shape) == 1:  # Bias or layernorm
                num_weights = W.shape[0]
                ssne_prob = ssne_probabilities[i] * 0.04  # Low probability of mutation here

                if random.random() < ssne_prob:
                    num_mutations = random.randint(0, int(math.ceil(
                        num_mutation_frac * num_weights)))  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim = random.randint(0, W.shape[0] - 1)
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
                        elif random_num < reset_prob:  # Reset probability
                            W[ind_dim] = random.gauss(0, 1)
                        else:  # mutauion even normal
                            W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

                        # Regularization hard limit
                        # W[ind_dim] = self.regularize_weight(W[ind_dim], self.args.weight_magnitude_limit)

    # 变异
    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            select_range = [x for x in range(self.size)]
            select_range.remove(i)
            r0, r1, r2 = np.random.choice(select_range, 3, replace=False)

            tmp = deepcopy(self.individuality[0])
            for tmp_param, r0_param, r1_param, r2_param in zip(tmp.parameters(),
                                                               self.individuality[r0].parameters(),
                                                               self.individuality[r1].parameters(),
                                                               self.individuality[r2].parameters()):
                tmp_param.set_value(r0_param + (r1_param - r2_param) * self.factor)
                # tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.factor
                paddle.where(self.min_range <= tmp_param <= self.max_range, tmp_param,
                             random.uniform(self.min_range, self.max_range))
                # for t in range(self.dimension):
                #     if tmp[t] > self.max_range or tmp[t] < self.min_range:
                #         tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)

    # 交叉选择
    def crossover_and_select(self):
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            # 随机替换网络参数中数值
            x = self.mutant[i].state_dict()

            for j in range(self.dimension):
                if random.random() > self.cr and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]

                reward = self.objective_function(self.mutant[i])
                # 超过原值则更新
                if reward > self.object_function_values[i]:
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = reward

    def get_best(self):
        m = max(self.object_function_values)
        i = self.object_function_values.index(m)
        print("轮数：" + str(self.cur_round))
        print("最佳个体：" + str(self.individuality[i]))
        print("目标函数值：" + str(m))
        self.cur_round = self.cur_round + 1
        return self.individuality[i]

    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            best_policy = self.get_best()
        return best_policy
