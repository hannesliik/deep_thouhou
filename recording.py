import numpy as np
import os
import pickle
import torch

class Recording:

    def __init__(self, experience, rewards: list, actions: list, q_scores: list, counter=0):
        self.counter = counter  # index of last frame (if the whole array is not used)
        self.experience = experience  # numpy array [t, RGB, w, h] of frames
        self.rewards = rewards  # rewards received at each time step
        self.actions = actions  # action taken at each time step
        self.q_scores = q_scores  # The predicted future reward (Q) for each possible action

    def get_current_state(self, frames_memory: int):
        shape = self.experience.shape
        return np.reshape(self.experience[self.counter - frames_memory: self.counter],
                          (3 * frames_memory, shape[2], shape[3],))

    def reshape_timesteps_to_channels(self, state_with_time_channel):
        """Concatenates the memory (timesteps) channels to RGB channels). A helper function"""
        shape_orig = state_with_time_channel.shape
        return np.reshape(state_with_time_channel, (shape_orig[0], shape_orig[1] * shape_orig[2], shape_orig[3], shape_orig[4]))

    def state_ready(self, frames_memory: int):
        return self.counter >= frames_memory

    def save(self, path_root: str):
        files = os.listdir(path_root)
        max_num = -1
        for file in files:
            num = file.split('_')[0]
            if num.isnumeric() and int(num) > max_num:
                max_num = int(num)
        file = open(os.path.join(path_root, str(max_num + 1) + '_recording.pickle'), 'wb')
        pickle.dump(self, file)
        file.close()

    def sample_random_transitions(self, batch_size, state_memory_size: int):
        max_index = self.counter - state_memory_size - 1
        ix = np.random.permutation(max_index)[:batch_size]
        num_transitions = len(ix)
        states = np.zeros([num_transitions, state_memory_size] + list(self.experience.shape[1:]))
        next_states = np.zeros([num_transitions, state_memory_size] + list(self.experience.shape[1:]))
        actions = []
        rewards = []
        q_scores_1 = []
        q_scores_2 = []
        for i, start_index in enumerate(ix):
            states[i, :, :, :, :] = self.experience[start_index:start_index + state_memory_size, :, :, :]
            next_states[i, :, :, :, :] = self.experience[start_index + 1:start_index + state_memory_size + 1, :, :, :]
            rewards.append(self.rewards[start_index])
            actions.append(self.actions[start_index])
            q_scores_1.append((self.q_scores[start_index]))
            q_scores_2.append((self.q_scores[start_index + 1]))
        return states, actions, next_states, rewards, q_scores_1, q_scores_2

def create_recording(frames_memory, monitor):
    # Create an array to hold the upcoming frames
    experience = np.zeros([frames_memory, 3, monitor['height'], monitor['width']],dtype=np.uint8)
    rewards = []
    actions = []
    q_scores = []
    return Recording(experience, rewards, actions, q_scores, counter=0)


def get_transition(recording: Recording, index: int, state_memory_size: int):
    """Given a history of frames or ':recording', return a single experience that is
    :state_memory_size frames long"""
    state = recording.experience[index:index + state_memory_size, :, :]
    next_state = recording.experience[index + 1:index + state_memory_size + 1, :, :]
    reward = recording.rewards[index + 1]
    action = recording.actions[index]
    q_1 = recording.q_scores[index]
    q_2 = recording.q_scores[index + 1]
    return state, action, reward, next_state, q_1, q_2



def get_current_state(recording: Recording, frames_memory: int):
    return recording.experience[recording.counter - frames_memory:recording.counter]
