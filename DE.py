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
