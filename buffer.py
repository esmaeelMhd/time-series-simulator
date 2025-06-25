import numpy as np

from collections import deque
from random import randint
from rlrd_util import collate

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # Select the number of batch_size indices from 0 to max_mem
        # It returns an array of shape (batch_size) with all random indices
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class TrajMemoryNoHidden:
    keep_reset_transitions: int = 0

    def __init__(self, memory_size, batchsize, device, history=1, remove_size=100):
        self.device = device
        self.batchsize = batchsize
        self.capacity = memory_size
        self.memory = []  # list is much faster to index than deque for big sizes
        # history: act_buf_size = act_delay + obs_delay
        self.history = deque(maxlen=history + 1)
        self.remove_size = remove_size

    def append(self, r, done, info, obs, action):
        self.history.append((r, obs, action))
        if not self.keep_reset_transitions and (info.get('TimeLimit.truncated', False) or info.get('reset', False)):
            self.history.clear()

        if len(self.history) == self.history.maxlen:
            (_, *r), m, a = zip(*self.history)
            self.memory.append((m, a, r, done))

        if done:
            self.history.clear()

        # remove old entries if necessary (delete generously so we don't have to do it often)
        if len(self.memory) > self.capacity:
            del self.memory[:self.capacity // self.remove_size + 1]

        return self

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def sample_indices(self):
        # doesn't return a list of random indices but rather a generator object 
        # that can produce these indices one at a time when iterated over
        return (randint(0, len(self.memory) - 1) for _ in range(self.batchsize))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self.memory[idx] for idx in indices]
        batch = collate(batch, self.device)
        return batch