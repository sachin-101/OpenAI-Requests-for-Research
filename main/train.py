import os
import sys
from os.path import abspath, dirname

import random
import time
import datetime
from collections import deque
import numpy as np

sys.path.append(dirname(dirname(__file__)))
from agent.Agent import DeepQ_agent
from env.Environment import Env


# Initialise the environment
max_env_width, max_env_height = 30, 30
env_width, env_height = 20, 20
display_width, display_height = 600, 600
agent_vision = 4
env = Env(max_env_width, max_env_height, env_width, env_height, display_width, display_height, agent_vision)

# Hyperparams ! all the magic happens here
HIDDEN_UNITS = (64, 32)
NETWORK_LR = 0.01
BATCH_SIZE = 64
UPDATE_EVERY = 5
GAMMA = 0.95
eps, eps_min, eps_decay = 1, 0.05, 0.9997
NUM_EPISODES = 10000    #number of episodes to train

parent_dir = 'Training Files'
i = 0
while True:
    train_dir = os.path.join(parent_dir, f'Training_{i}')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        break
    i += 1

#directory where tensorflow will store the logs, for tensorboard to read  
logdir_agent_1 = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "p1")
logdir_agent_2 = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "p2")

# Initialise the agents
agent1 = DeepQ_agent(env, HIDDEN_UNITS, NETWORK_LR, BATCH_SIZE, UPDATE_EVERY, GAMMA, logdir=logdir_agent_1)
agent2 = DeepQ_agent(env,  HIDDEN_UNITS, NETWORK_LR, BATCH_SIZE, UPDATE_EVERY, GAMMA, logdir=logdir_agent_2)

#---------------Let's Train the agents-------------------------#
        
        
scores1, scores2 = [], []
stats_1, stats_2 = [0, 0, 0, 0], [0, 0, 0, 0]
INCREASE_EVERY, SAVE_EVERY = 500, 100
scores_window1, scores_window2 = deque(maxlen=INCREASE_EVERY), deque(maxlen=INCREASE_EVERY)
#loop over episodes
for i_episode in range(1, NUM_EPISODES+1):
    
    eps = max(eps*eps_decay, eps_min)
    state_1, vision_1, state_2, vision_2 = env.reset()
    a1 = agent1.act(state_1, eps)
    a2 = agent2.act(state_2, eps)
    #a1 = int(input())
    #a2 = int(input())

    #render the environment         
    env.render((a1, a2), (vision_1, vision_2), episode=i_episode, epsilon=eps, gamma=GAMMA, stats=(stats_1, stats_2), train=True)
    score1, score2 = 0, 0

    while True:
        
        next_state_1, reward_1, done_1, vision_1, info_1 = env.step(a1, player=1)
        next_state_2, reward_2, done_2, vision_2, info_2 = env.step(a2, player=2)

        #add the experience to agent's memory
        agent1.add_experience(state_1, a1, reward_1, next_state_1, done_1)
        agent1.learn()
        
        agent2.add_experience(state_2, a2, reward_2, next_state_2, done_2)
        agent2.learn()

        #render the environment
        env.render((a1, a2), (vision_1, vision_2), episode=i_episode, epsilon=eps, gamma=GAMMA, stats=(stats_1, stats_2), train=True)

        score1 += reward_1
        score2 += reward_2

        # update stats
        stats_1[info_1] += 1
        stats_2[info_2] += 1

        if done_1 or done_2:
            #time.sleep(2)
            break
        
        #update state and action
        state_1, state_2 = next_state_1, next_state_2
        a1 = agent1.act(state_1, eps)
        a2 = agent2.act(state_2, eps)
        #a1 = int(input())
        #a2 = int(input())

    scores_window1.append(score1)
    scores_window2.append(score2)
    scores1.append(score1)
    scores1.append(score2)
  
    print('episode', i_episode)
    # monitor progress    
    if (i_episode + 1)% SAVE_EVERY == 0:
        agent1.save(train_dir, i_episode+1, info='p1')
        agent2.save(train_dir, i_episode+1, info='p2')
        
        print('\rEpisode {}\t Score_1 {}\tAvg Score_1: {:.2f}\t Score_2 {}\tAvg Score_2: {:.2f}'\
            .format(i_episode+1, score1, np.mean(scores_window1), score2, np.mean(scores_window2)))
        print('stats Player 1', stats_1)
        print('stats Player 2', stats_2)
        sys.stdout.flush()
    
    #increase environment size
    if (i_episode +1)% INCREASE_EVERY == 0:
        env.change_size(1, 1)  #increase the env size by 1
    
    #after 6k episodes increase up the training process
    if (i_episode + 1) == 6000:
        INCREASE_EVERY = 100  #That is increase the size by 1, every 100 episodes
        SAVE_EVERY = 100      

#save the agent's q-network for testing
agent1.save(train_dir, 'final', 'p1')
agent2.save(train_dir, 'final', 'p2')
        