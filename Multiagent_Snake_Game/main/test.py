import cv2
import sys
import time
import os
import pickle
from os.path import abspath, dirname

sys.path.append(dirname(dirname(__file__)))
from agent.Agent import DeepQ_agent
from env.Environment import Env


parent_dir = 'Training Files'
i = 0
while True:
    train_dir = os.path.join(parent_dir, f'Training_{i}')
    if not os.path.exists(train_dir):
        break
    if len(os.listdir(train_dir)) > 0:
        curr_dir = train_dir
    i += 1

models = os.listdir(curr_dir)
latest_model = 0
for m in models:
    num = int(m.split('_')[1])
    if num > latest_model:
        latest_model = num
    

curr_dir = "Training Files/Training_9"
with open(f'{curr_dir}/params.pickle', 'rb') as f:
    params = pickle.load(f)


#creating the environment
env = Env(params['max_env_width'], params['max_env_height'], 
        params['env_width'], params['env_height'], 
        params['display_width'], params['display_height'], 
        params['agent_vision'])

agent1 = DeepQ_agent(env, hidden_units=params['hidden_units'], summarry=True)
agent2 = DeepQ_agent(env, hidden_units=params['hidden_units'], summarry=False)

model1_dir = f'{curr_dir}/model_{latest_model}_p1.pth.tar'
model2_dir = f'{curr_dir}/model_{latest_model}_p2.pth.tar'
print('Loading Models')
print(model1_dir)
print(model2_dir)
print("-"*10)

agent1.load_model(model1_dir)
agent2.load_model(model2_dir)

NUM_TIMES = 20

stats1, stats2 = [0, 0, 0, 0], [0, 0, 0, 0]

#testing the agents
for i in range(NUM_TIMES):  #running for 10 times
    
    state_1, vision_1, state_2, vision_2 = env.reset()
    a1 = agent1.act(state_1)
    a2 = agent2.act(state_2)
    
    env.render((a1, a2), (vision_1, vision_2), (stats1, stats2))
    
    score1, score2 = 0, 0
    total1, total2 = 0, 0
    while True:
        
        next_state_1, reward_1, done_1, vision_1, info_1 = env.step(a1, player=1)
        next_state_2, reward_2, done_2, vision_2, info_2 = env.step(a2, player=2)

        #render the environment
        env.render((a1, a2), (vision_1, vision_2), (stats1, stats2))
        time.sleep(0.2)

        score1 += reward_1
        score2 += reward_2

        # update stats
        stats1[info_1] += 1
        stats2[info_2] += 1

        if done_1 or done_2:
            time.sleep(2)
            print('1', total1, '2', total2)
            break
        