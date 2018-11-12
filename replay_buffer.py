import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, length, state_shape, n_frames_obs):
        self.length = length
        self.state_shape = state_shape
        self.buffer = np.zeros([length] + state_shape)
        self.rewards = np.zeros((length))
        self.actions = np.zeros((length))
        self.ix = 0
        self.fill_ix = 0  # How much has been filled
        self.frame_buffer = deque(maxlen=n_frames_obs)
        self.waiting_for_effect = False

    def _add_frame(self, frame):
        self.ix = (self.ix + 1) % self.length
        self.fill_ix = min(self.length, self.fill_ix + 1)
        self.buffer[self.ix] = frame

    def push_frame(self, frame):
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) == self.frame_buffer.maxlen:
            self.add_frame(np.array(self.frame_buffer))
            self.waiting_for_effect = True

    def buffer_init(self):
        return len(self.frame_buffer) == self.frame_buffer.maxlen

    def encode_last_frame(self):
        return np.array(self.frame_buffer)

    def add_effects(self, action, reward, ix=None):
        if ix is None:
            ix = self.ix
        self.rewards[ix] = reward
        self.actions[ix] = action
        self.waiting_for_effect = False

    def sample(self, batch_size):
        sample_ix = np.random.permutation(self.fill_ix - 1)[:batch_size]
        return self.buffer[sample_ix], self.actions[sample_ix], self.rewards[sample_ix], self.buffer[sample_ix + 1]

    def can_sample(self, batch_size):
        return self.fill_ix >= batch_size
