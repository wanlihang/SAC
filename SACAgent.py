import numpy as np
import paddle


class SACAgent:
    def __init__(self, algorithm):
        self.alg = algorithm
        self.alg.soft_update_target(decay=0)

    def sample(self, state):
        state = np.array(state)
        state = paddle.to_tensor(state.reshape(1, -1), dtype='float32')
        action, _ = self.alg.sample(state)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, state, action, reward, next_state):
        reward = np.expand_dims(reward, -1)
        state = np.array(state)
        next_state = np.array(next_state)
        state = paddle.to_tensor(state, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_state = paddle.to_tensor(next_state, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(state, action, reward, next_state)
        return critic_loss, actor_loss
