from collections import deque
import random
import numpy as np
import torch


class ReplayMemory:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.BATCH_SIZE = batch_size
        
    
    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory'''
        e = tuple((state, action, reward, next_state, done))
        self.memory.append(e)
        
    
    def sample(self, state_shape, device):
        '''Randomly sample a batch of experiences from memory'''
        
        experiences = random.sample(self.memory, k=self.BATCH_SIZE)
        #extracting the SARSA
        states, actions, rewards, next_states, dones = zip(*experiences)

        #converting them to torch tensors for easy operations
        states = torch.tensor(states).reshape(-1, state_shape).to(device, dtype=torch.float32)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(device, dtype=torch.float)
        print(type(next_states[0]))
        print(next_states[0].shape)
        print(state_shape)
        print(len(states))
        print(states[0].shape)
        try:
            next_states = torch.tensor(next_states).reshape(-1, state_shape).to(device, dtype=torch.float32)
        except ValueError:
            print('\n Analysis')
            print(type(next_states))
            print(len(next_states))
            print()
            print('state_shape', state_shape)
        dones = torch.tensor(dones).unsqueeze(1).to(device, dtype=torch.float)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):     
        '''overriding the __len___ method'''
        return len(self.memory)