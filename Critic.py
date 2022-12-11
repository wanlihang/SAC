import paddle
import paddle.nn.functional as F
from paddle import nn


class Critic(paddle.nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        for m in self.__module__:
            if isinstance(m, nn.Linear):
                nn.initializer.XavierUniform(m.weight)

        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = paddle.concat([state, action], 1)

        # Q1
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
