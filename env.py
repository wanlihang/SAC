import copy
import queue
import time

import numpy as np

import node
import task


def is_ready(node_load, task_cost):
    for index, value in enumerate(node_load):
        if value + task_cost[index] > 100.0:
            return False
    return True


def add_load(node_load, task_cost):
    result = []
    for index, value in enumerate(node_load):
        result.append(value + task_cost[index])
    return result


def remove_load(node_load, task_cost):
    result = []
    for index, value in enumerate(node_load):
        result.append(value - task_cost[index])
    return result


class Environment:
    def __init__(self):
        # 环境初始化时的固定值：节点列表、任务列表、任务代价列表
        self.NODE_LIST = ["node1", "node2", "node3"]
        self.TASK_LIST = [4, 7, 8, 9, 13, 14, 17, 20, 22, 25, 27]
        self.task_cost_map = task.init_task_cost_map()

        # 环境初始化时的变量：开始时间、排队队列、队列长度、执行队列、等待队列
        self.node_exec_priority_queue = {}
        self.node_wait_queue = {}
        for node_name in self.NODE_LIST:
            # 执行队列，字典类，每个节点一个队列（小根堆）
            self.node_exec_priority_queue[node_name] = queue.PriorityQueue()
            # 等待队列，字典类，每个节点一个队列（顺序队列）
            self.node_wait_queue[node_name] = queue.Queue()
        self.BEGIN_TIME = None
        self.queue_queue = queue.PriorityQueue()
        self.request_num = 0

        # 输入到神经网络中的环境状态变量
        self.task_arrive_time = None
        self.task_id = None
        self.node_load = node.get_node_load(self.NODE_LIST)
        self.node_task_cost = {}
        self.node_wait_time = {}

    def reset(self):
        # 环境初始化时的变量：开始时间、排队队列、队列长度、执行队列、等待队列
        self.node_exec_priority_queue = {}
        self.node_wait_queue = {}
        for node_name in self.NODE_LIST:
            # 执行队列，字典类，每个节点一个队列（小根堆）
            self.node_exec_priority_queue[node_name] = queue.PriorityQueue()
            self.node_exec_priority_queue[node_name].queue.clear()
            # 等待队列，字典类，每个节点一个队列（顺序队列）
            self.node_wait_queue[node_name] = queue.Queue()
            self.node_wait_queue[node_name].queue.clear()
        self.BEGIN_TIME = time.time()
        self.queue_queue = queue.PriorityQueue()
        self.queue_queue.queue.clear()
        arrive_time_list = np.random.poisson(1000, 176)
        # arrive_time_list.sort()
        for index, value in enumerate(arrive_time_list):
            self.queue_queue.put((value / 1000.0, self.TASK_LIST[index % len(self.TASK_LIST)]))
        self.request_num = self.queue_queue.qsize()

        # 输入到神经网络中的环境状态变量
        self.task_arrive_time = None
        self.task_id = None
        self.node_load = node.get_node_load(self.NODE_LIST)
        self.node_task_cost = {}
        self.node_wait_time = {}

        # 更新输入到神经网络中的环境状态变量
        state = self.update_state()
        return state
        # return state / np.linalg.norm(state)

    def step(self, action):
        # 动作转换
        if action <= - 1.0 / 3:
            action = 0
        elif action <= 1.0 / 3:
            action = 1
        else:
            action = 2

        node_name = self.NODE_LIST[action]

        # 获取奖励
        reward = self.get_reward(node_name)

        arrive_time, task_id = self.queue_queue.get()
        task_cost = self.task_cost_map[node_name][task_id]

        # 异常情况：空节点全部资源都不足以支持当前任务完成
        # if self.node_exec_priority_queue[node_name].empty() and not is_ready(self.node_load[node_name], task_cost):
        #     print('[env] [step] [error: exec list is empty but resources is not enough]')
        #     return [], float(), self.queue_queue.empty()

        # 等待队列为空且资源充足时，直接将任务加入执行队列
        if self.node_wait_queue[node_name].empty() and is_ready(self.node_load[node_name], task_cost):
            self.node_exec_priority_queue[node_name].put((arrive_time + task_cost[-1] + self.node_wait_time[node_name],
                                                          self.node_wait_time[node_name], task_cost))
            self.node_load[node_name] = add_load(self.node_load[node_name], task_cost)

        # 等待队列不为空时，调整执行队列和等待队列
        else:
            # 1. 将已经执行完的执行队列任务出队
            while True:
                if self.node_exec_priority_queue[node_name].empty():
                    break
                exec_end_time, exec_wait_time, exec_task_cost = self.node_exec_priority_queue[node_name].queue[0]
                if exec_end_time >= arrive_time:
                    break
                # 出队，减少负载
                self.node_exec_priority_queue[node_name].get()
                self.node_load[node_name] = remove_load(self.node_load[node_name], exec_task_cost)

            # 2. 将已经执行完的等待队列任务出队
            while True:
                if self.node_wait_queue[node_name].empty():
                    break
                wait_end_time, wait_wait_time, wait_task_cost = self.node_wait_queue[node_name].queue[0]
                if wait_end_time >= arrive_time:
                    break
                self.node_wait_queue[node_name].get()

            # 3. 将到达任务放入排队队列
            self.node_wait_queue[node_name].put((arrive_time + task_cost[-1] + self.node_wait_time[node_name],
                                                 self.node_wait_time[node_name], task_cost))

            # 4. 当等待队列不为空时，且资源充足时，将等待队列不断出队加入到执行队列中
            while True:
                if self.node_wait_queue[node_name].empty():
                    break
                wait_end_time, wait_wait_time, wait_task_cost = self.node_wait_queue[node_name].queue[0]
                if not is_ready(self.node_load[node_name], wait_task_cost):
                    break
                # 等待队列出队到执行队列，增加负载
                self.node_exec_priority_queue[node_name].put(self.node_wait_queue[node_name].get())
                self.node_load[node_name] = add_load(self.node_load[node_name], wait_task_cost)

        # 判断是否结束
        done = self.queue_queue.empty()
        state = []
        if not done:
            # 更新环境状态
            state = self.update_state()
            # state /= np.linalg.norm(state)
        return state, reward, done

    # 标准化环境状态空间
    def update_state(self):
        # 环境状态1：根据排队队列生成的唯一id
        state = []
        # state.append(self.request_num - self.queue_queue.qsize())

        self.task_arrive_time, self.task_id = self.queue_queue.queue[0]
        # 环境状态2：队首任务到达时间、id
        # state.append(self.task_arrive_time)
        # state.append(self.task_id)

        for node_name in self.NODE_LIST:
            # 环境状态3：每个节点的负载数据
            state.extend(self.node_load[node_name])

        for node_name in self.NODE_LIST:
            self.node_task_cost[node_name] = self.task_cost_map[node_name][self.task_id]
            # # 环境状态4：任务在每个节点上运行需要的资源消耗数据
            state.extend(self.node_task_cost[node_name])

            # 环境状态4：任务在每个节点上运行时间
            # state.append(self.node_task_cost[node_name][-1])

        for node_name in self.NODE_LIST:
            self.node_wait_time[node_name] = self.get_wait_time(node_name)
            # 环境状态5：任务在每个节点上需要等待的时间
            state.append(self.node_wait_time[node_name])

        return state

    # 计算任务在当前节点上的等待时间
    def get_wait_time(self, node_name):
        # 深拷贝 当前节点负载、当前任务资源消耗、当前执行队列、当前等待队列 数据
        current_node_load = copy.deepcopy(self.node_load[node_name])
        current_task_cost = copy.deepcopy(self.node_task_cost[node_name])
        current_exec_priority_queue = queue.PriorityQueue()
        current_exec_priority_queue.queue.clear()
        for item in self.node_exec_priority_queue[node_name].queue:
            current_exec_priority_queue.put(item)
        current_wait_queue = queue.Queue()
        current_wait_queue.queue.clear()
        for item in self.node_wait_queue[node_name].queue:
            current_wait_queue.put(item)

        # 等待队列为空且资源充足时，任务无需等待
        if current_wait_queue.empty() and is_ready(current_node_load, current_task_cost):
            return float()

        # 若节点资源不足，需要先清空等待队列，再计算等待时间
        wait_time = float()
        # 清空等待队列
        while not current_wait_queue.empty():
            wait_end_time, wait_wait_time, wait_task_cost = current_wait_queue.queue[0]
            # 执行队列出队
            while not is_ready(current_node_load, wait_task_cost):
                exec_end_time, exec_wait_time, exec_task_cost = current_exec_priority_queue.get()
                current_node_load = remove_load(current_node_load, exec_task_cost)
                wait_time = exec_end_time - self.task_arrive_time
            # 等待队列出队到执行队列
            current_exec_priority_queue.put(current_wait_queue.get())
            current_node_load = add_load(current_node_load, wait_task_cost)
        # 计算当前任务等待时间
        while not is_ready(current_node_load, current_task_cost):
            exec_end_time, exec_wait_time, exec_task_cost = current_exec_priority_queue.get()
            current_node_load = remove_load(current_node_load, exec_task_cost)
            wait_time = exec_end_time - self.task_arrive_time
        return wait_time

    # 获取奖励
    def get_reward(self, node_name):
        reward = [1.0, 0.3, -0.7]
        wait_time_list = list(self.node_wait_time.values())
        wait_time_list.sort()
        wait_time = self.node_wait_time[node_name]
        index = wait_time_list.index(wait_time)
        exec_time = self.node_task_cost[node_name][-1]
        # return reward[index]
        return - wait_time / 1000