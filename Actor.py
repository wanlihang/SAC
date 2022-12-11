import paddle
import paddle.nn.functional as F
from paddle import nn


class Actor(paddle.nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        for m in self.__module__:
            if isinstance(m, nn.Linear):
                nn.initializer.XavierUniform(m.weight)

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.std_linear = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        act_mean = self.mean_linear(x)
        act_std = self.std_linear(x)
        act_log_std = paddle.clip(act_std, -1, 1)
        return act_mean, act_log_std
