import numpy as np


class ReplayBuffer:
    def __init__(self, length, state_shape):
        self.length = length
        self.state_shape = state_shape
        self.buffer = np.zeros([length] + state_shape)
        self.rewards = np.zeros((length))
        self.actions = np.zeros((length))
        self.ix = 0
        self.fill_ix = 0  # How much has been filled

    def add_frame(self, frame):
        self.ix = (self.ix + 1) % self.length
        self.fill_ix = min(self.length, self.fill_ix + 1)
        self.buffer[self.ix] = frame
        return self.ix

    def add_effects(self, action, reward, ix=None):
        if ix is None:
            ix = self.ix
        self.rewards[ix] = reward
        self.actions[ix] = action

    def sample(self, batch_size):
        sample_ix = np.random.permutation(self.fill_ix)[:batch_size]
        return self.buffer[sample_ix], self.actions[sample_ix], self.rewards[sample_ix]
