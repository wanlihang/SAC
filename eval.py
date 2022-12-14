import sys

import paddle
from paddle.distribution import Normal

from Actor import Actor


def get_action(state):
    # print(state)
    return
    # 初始化模型
    actor = Actor(30, 1)
    # 载入模型参数
    actor_state_dict = paddle.load("linear_net.pdparams")
    # 将load后的参数与模型关联起来
    actor.set_state_dict(actor_state_dict)
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


if __name__ == '__main__':
    argv_list = []
    for i in range(1, len(sys.argv)):
        argv_list.append(sys.argv[i])
        print(sys.argv[i])
    get_action(argv_list[0])
