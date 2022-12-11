import numpy as np


class Memory(object):
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = np.zeros((max_size, state_dim), dtype='float32')
        self.action = np.zeros((max_size, action_dim), dtype='float32')
        self.reward = np.zeros((max_size,), dtype='float32')
        self.next_state = np.zeros((max_size, state_dim), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    # 抽样指定数量（batch_size）的经验
    def sample(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        state = self.state[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_state = self.next_state[batch_idx]
        return state, action, reward, next_state

    def append(self, state, action, reward, next_state):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.state[self._curr_pos] = state
        self.action[self._curr_pos] = action
        self.reward[self._curr_pos] = reward
        self.next_state[self._curr_pos] = next_state
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size
