import copy

import paddle
import paddle.nn.functional as F
from paddle.distribution import Normal


class SAC:
    def __init__(self, model, gamma=None, tau=None, alpha=None, actor_lr=None, critic_lr=None):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def sample(self, state):
        act_mean, act_log_std = self.model.policy(state)
        normal = Normal(act_mean, act_log_std.exp())
        # 重参数化  (mean + std*N(0,1))
        x_t = normal.sample([1])
        action = paddle.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
        log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)
        return action[0], log_prob[0]

    def save(self):
        # 保存Layer参数
        paddle.save(self.model.actor_model.state_dict(), "linear_net.pdparams")

    def learn(self, state, action, reward, next_state):
        critic_loss = self._critic_learn(state, action, reward, next_state)
        actor_loss = self._actor_learn(state)

        self.soft_update_target()
        return critic_loss, actor_loss

    def _critic_learn(self, state, action, reward, next_state):
        with paddle.no_grad():
            next_action, next_log_pro = self.sample(next_state)
            q1_next, q2_next = self.target_model.value(next_state, next_action)
            min_q_next = paddle.minimum(q1_next, q2_next) - self.alpha * next_log_pro
            target_q = reward + self.gamma * min_q_next

        cur_q1, cur_q2 = self.model.value(state, action)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, state):
        action, log_pi = self.sample(state)
        q1_pi, q2_pi = self.model.value(state, action)
        min_q_pi = paddle.minimum(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def soft_update_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.soft_update(self.target_model, decay=decay)

    def get_actor(self):
        return self.model.get_actor()

    def hard_update_target(self, actor):
        self.model.hard_update(self.target_model, actor)
