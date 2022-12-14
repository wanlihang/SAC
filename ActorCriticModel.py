from copy import deepcopy

import paddle

from Actor import Actor
from Critic import Critic


class ActorCriticModel(paddle.nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticModel, self).__init__()
        self.actor_model = Actor(state_dim, action_dim)
        self.critic_model = Critic(state_dim, action_dim)

    def policy(self, state):
        return self.actor_model(state)

    def value(self, state, action):
        return self.critic_model(state, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def soft_update(self, target_model, decay=0.0):
        target_vars = dict(target_model.named_parameters())
        for name, var in self.named_parameters():
            target_data = decay * target_vars[name] + (1 - decay) * var
            target_vars[name] = target_data
        target_model.set_state_dict(target_vars)

    def get_actor(self):
        return deepcopy(self.actor_model)

    def hard_update(self, target_model, actor):
        for target_param, param in zip(target_model.parameters(), actor.parameters()):
            target_param.set_value(param)
