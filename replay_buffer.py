import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, length, state_shape, n_frames_obs, baseline_priority=1, gamma=0.9, reward_steps=1):
        self.length = length
        self.state_shape = state_shape
        self.buffer = np.zeros([length] + list(state_shape))
        self.rewards = np.zeros((length))
        self.actions = np.zeros((length))
        self.priorities = np.zeros((length))
        self.ix = 0
        self.fill_ix = 0  # How much has been filled
        self.frame_buffer = deque(maxlen=n_frames_obs)
        self.waiting_for_effect = False
        self.baseline_priority = baseline_priority
        self.gamma = gamma
        self.indexes_buffer = deque(maxlen=reward_steps)
        self.rewards_buffer = deque(maxlen=reward_steps)
        self.actions_buffer = deque(maxlen=reward_steps)

    def _add_frame(self, frame):
        self.ix = (self.ix + 1) % self.length
        self.fill_ix = min(self.length, self.fill_ix + 1)

        # Concatenate previous frames into the colors channel
        shape_orig = frame.shape
        self.buffer[self.ix] = np.reshape(frame,
                                          (shape_orig[0] * shape_orig[1], shape_orig[2], shape_orig[3]))
        return self.ix

    def push_frame(self, frame):
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) == self.frame_buffer.maxlen:
            ix = self._add_frame(np.array(self.frame_buffer))
            self.waiting_for_effect = True
            return ix
        return None

    def buffer_init(self):
        return len(self.frame_buffer) == self.frame_buffer.maxlen

    def encode_last_frame(self):
        return np.reshape(np.array(self.frame_buffer), self.buffer.shape[1:])

    def add_effects(self, new_action, new_reward, new_ix=None):
        if new_ix is None:
            new_ix = self.ix

        # apply rewards to buffer
        for i in range(len(self.rewards_buffer)):
            self.rewards_buffer[i] += self.gamma ** (i + 1) * new_reward
        is_full = len(self.rewards_buffer) == self.rewards_buffer.maxlen
        if is_full:
            new_ix = self.indexes_buffer.pop()
            rew = self.rewards_buffer.pop()
            action = self.actions_buffer.pop()
            self.rewards[new_ix] = rew
            self.actions[new_ix] = action
            self.priorities[new_ix] = 0
            print(rew)
        self.indexes_buffer.appendleft(new_ix)
        self.rewards_buffer.appendleft(new_reward)
        self.actions_buffer.appendleft(new_action)

    @staticmethod
    def _abs_softmax(x):
        x = np.abs(x)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def _norm_priorities(p, a=0.5):
        p_a = p ** a
        return p_a / (np.sum(p_a) + 1e-7)

    def add_errors(self, sample_ix, errors):
        self.priorities[sample_ix] = errors

    def sample(self, batch_size, weighted=True):
        max_ix = self.fill_ix - (1 if self.waiting_for_effect else 0)
        # print(p.shape, max_ix)
        if weighted:
            p = ReplayBuffer._norm_priorities(self.priorities[:max_ix] + self.baseline_priority)
            sample_ix = np.random.choice(max_ix, batch_size, p=p)
        else:
            sample_ix = np.random.choice(max_ix, batch_size)
        # sample_ix = np.random.permutation(self.fill_ix - 1)[:batch_size]
        return self.buffer[sample_ix], self.actions[sample_ix].astype(np.int64, copy=False), self.rewards[sample_ix], \
               self.buffer[sample_ix + 1], sample_ix

    def can_sample(self, batch_size):
        return self.fill_ix >= batch_size
