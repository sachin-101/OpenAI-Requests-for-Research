import time
import random
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F

from agent.q_network import QNetwork 
from agent.memory import ReplayMemory

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  
class DeepQ_agent:

    def __init__(self, env, hidden_units = None, network_LR = 0.001, batch_size = 64, 
                    update_every=4, gamma=1.0, summarry=True):
        self.env = env
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        memory_capacity = int(1e5)   #this is pythonic
        
        self.nA = env.ACTION_SPACE              #number of actions agent can perform
        self.UPDATE_EVERY = update_every

        #let's give it some brains
        self.qnetwork_local = QNetwork(self.env.STATE_SPACE, hidden_units, self.nA).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=network_LR)
        if summarry:
            print(self.qnetwork_local)
        
        #I call the target network as the PC
        # Where our agent stores all the concrete and important stuff
        self.qnetwork_target = QNetwork(self.env.STATE_SPACE, hidden_units, self.nA).to(device)

        #and the memory of course
        self.memory = ReplayMemory(memory_capacity, self.BATCH_SIZE) 

        #handy temp variable
        self.t = 0

#----------------------Learn from experience-----------------------------------#

    def learn(self, writer=None, episode=-1):
        '''
            hell yeah   
        '''

        if self.memory.__len__() > self.BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample(self.env.STATE_SPACE, device)

            # gather, refer here https://stackoverflow.com/a/54706716/10666315
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            
            # Get max predicted action-values for next states, using target model
            # detach, detaches the tensor from graph
            # max(1) return two tensors, one containing max values along dim=1, 
            # other containing indices of max values along dim=1
            # unsqueeze(1) inserts a dimension of size one at specified position
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

            Q_targets =  rewards + (self.GAMMA * Q_targets_next * (1 - dones))
            
            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if writer is not None:
                writer.add_scalar('Training Loss', loss, episode)  # write to tensorboard summary
                

            if self.t == self.UPDATE_EVERY:
                self.qnetwork_target.state_dict = self.qnetwork_local.state_dict  # update target network
                self.t = 0
            else:
                self.t += 1


#-----------------------Time to act-----------------------------------------------#

    def act(self, state, epsilon = 0):                 #set to NO exploration by default
        state = torch.from_numpy(state).to(device, dtype=torch.float32)
        action_values = self.qnetwork_local(state)    #returns a vector of size = self.nA
        if random.random() > epsilon:
            action = torch.argmax(action_values).item()      #choose best action - Exploitation
        else:
            action = random.randint(0, self.nA-1)  #choose random action - Exploration
        
        return action

#-----------------------------Add experience to agent's memory------------------------#

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

#---------------------helpful save function-------------------------------------#
    
    def save(self, dir, episode, info):
        torch.save(self.qnetwork_local.state_dict() , f'{dir}/model_{episode}_{info}.pth.tar')

#----------------------Load a saved model----------------------------------------#

    def load_model(self, model_path):
        self.qnetwork_local.state_dict = torch.load(model_path)