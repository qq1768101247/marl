from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.log_probs = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        
    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def __len__(self):
        return len(self.states)    