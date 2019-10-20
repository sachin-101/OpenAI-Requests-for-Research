import random
from PIL import Image
import cv2
import numpy as np
import time
from collections import deque

from snake.snake import Snake
from env.display_utils import add_info, add_states

"""
    ___     __               __ 
    |__ | | |_| \  / | \  /  |_
    __| |_| |\   \/  |  \/   |_   

"""

#colours - b, g, r
WHITE = (255,255,255)
BLUE = (255,0,0)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
ORANGE = (0,165,255)
FOOD_COLOR = ORANGE
BOUNDARY_COLOR = BLACK
BACKGROUND_COLOR = np.array(WHITE)

# action --> chage in position (x, y)
ACTION_RESULT = {0:(-1,0), #left
                 1:(1, 0), #right
                 2:(0,-1), # up
                 3:(0,1)}  # down   

PLAYER_COLORS = {1:GREEN,
                2:RED}
PLAYER_HEAD_COLOR = BLUE

CONSTANTS = {'boundary':-5, 'p1':1, 'p1_head':2, 'background':0, 'p2':-1, 'p2_head':-2, 'food':5}

class Env():

    def __init__(self, max_width, max_height, init_width, init_height, display_width, display_height, agent_vision):

        
        self.max_w = max_width
        self.max_h = max_height
        self.w = init_width
        self.h = init_height
        self.disp_w, self.disp_h = display_width, display_height
        
        self.ACTION_SPACE = 4
        self.agent_vision = agent_vision
        self.STATE_SPACE = (agent_vision*2)**2 + 2 # agent_vision*2)
                
#-------Increase the size of environment------#
    
    def change_size(self, w_change, h_change):
        self.w = min(self.w + w_change, self.max_w-2)
        self.h = min(self.h + h_change, self.max_h-2)

#---------helpful function to get boundaries----------#

    def get_boundaries(self):
        odd_w, odd_h = self.w%2, self.h%2
        mid_width, mid_height = int(self.max_w/2), int(self.max_h/2)
        x1, x2 = mid_width-int(self.w/2), mid_height-1+int(self.w/2)+odd_w
        y1, y2 = mid_width-int(self.h/2), mid_height-1+int(self.h/2)+odd_h
        return x1, x2, y1, y2

#-------------Returns random positions within the boundary--------------#

    def get_randoms(self, length = 1):

        x1, x2, y1, y2 = self.get_boundaries() 
        #logic to keep the snake AI inside the boundary
        if length>1:
            max_w, max_h = self.max_w-length, self.max_h-length
            if x1<length-1: x1 = length-1
            if x2>max_w: x2 = max_w
            if y1<length-1: y1 = length-1
            if y2>max_h: y2 = max_h    
        a = random.randint(x1, x2)
        b = random.randint(y1, y2)
        return a,b  


#-----Decide the playing region------#

    def draw_boundary(self, env):
        x1, x2, y1, y2 = self.get_boundaries()
        env[0:y1, :] = CONSTANTS['boundary']
        env[y2+1:,:] = CONSTANTS['boundary']
        env[y1:y2+1, 0:x1] = CONSTANTS['boundary']
        env[y1:y2+1, x2+1:] = CONSTANTS['boundary']
        return env

#-------Reset the environment-------#

    def reset(self):

        # reset env
        self.env = np.zeros((self.max_h, self.max_w))  # background
        self.env = self.draw_boundary(self.env)

        # Add player 1    
        head_x1, head_y1 = self.get_randoms()
        self.p1 = Snake(head_x1, head_y1 , self.w, self.h)
        self.env = self.p1.draw(self.env, player=1)

        # Add player 2    
        head_x2, head_y2 = self.get_randoms()
        self.p2 = Snake(head_x2, head_y2 , self.w, self.h)
        self.env = self.p2.draw(self.env, player=2)

        # Add food
        self.food_x, self.food_y = self.get_randoms()
        self.env[self.food_y, self.food_x] = CONSTANTS['food']
        
        # Get player's states (v for vision)
        self.v1 = self.p1.look(self.env, self.agent_vision)
        self.v2 = self.p2.look(self.env, self.agent_vision)
        
        return np.hstack((self.v1.ravel(), (head_x1 - self.food_x, head_y1 - self.food_y))), self.v1, \
               np.hstack((self.v2.ravel(), (head_x2 - self.food_x, head_y2 - self.food_y))), self.v2

#-----Render the environment---------#
    
    def render(self, actions, states, stats, train = True, episode=-1, epsilon=-1, gamma=-1):
        
        disp_matrix = self.get_display_matrix(self.env)
        state1, state2 = self.get_display_matrix(states[0]), self.get_display_matrix(states[1])
        disp_env = add_states(disp_matrix, state1, actions[0], left=True)
        disp_env = add_states(disp_env, state2, actions[1])
        
        # sprinkle some info to the disp
        train_params = None
        if train:
            train_params = { 'Episode': episode, 'Epsilon': epsilon, 'Gamma':gamma}
        p1_params = {'Player':1, 'stats':stats[0]}
        p2_params = {'Player':2, 'stats':stats[1]}
        img = add_info(self.disp_w, self.disp_h, disp_env, p1_params, p2_params, train_params=train_params)
        
        #img2 = Image.fromarray(self.env)
        #img2 = np.array(img2.resize(self.max_w, self.max_h))
        cv2.imshow("Sliterh_io", img)
        #cv2.imshow("AI v/s AI", self.env)
        #time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):  #when Q is pressed
            print('Stop Execution')
            cv2.destroyAllWindows()
            quit()

#------------Adds color to matrix------------------------------#
    
    @staticmethod
    def get_display_matrix(matrix):
        disp_matrix = np.ones((*matrix.shape, 3), dtype=np.uint8)*255  # background
        disp_matrix[np.where(matrix == CONSTANTS['boundary'])] = BOUNDARY_COLOR
        disp_matrix[np.where(matrix == CONSTANTS['p1'])] = PLAYER_COLORS[1]
        disp_matrix[np.where(matrix == CONSTANTS['p2'])] = PLAYER_COLORS[2]
        disp_matrix[np.where(matrix == CONSTANTS['p1_head'])] = PLAYER_HEAD_COLOR 
        disp_matrix[np.where(matrix == CONSTANTS['p2_head'])] = PLAYER_HEAD_COLOR
        disp_matrix[np.where(matrix == CONSTANTS['food'])] = FOOD_COLOR
        return disp_matrix

#----------Agent takes an action and environment changes--------#

    def step(self, action, player):
        
        info = 0
        # 1 - food eaten
        # 2 - Hit opponent
        # 3 - Died on Boundary
        p = self.p1 if player==1 else self.p2
        opp_player = 2 if player==1 else 1

        done, reward = False, -1  # positive reward

        x1, x2, y1, y2 = self.get_boundaries()
        
        x, y = ACTION_RESULT[action]
        head_x, head_y = p.head_pos()
        new_x, new_y = head_x + x, head_y + y     # new head positions
        
        # check outside boundary
        if new_x < x1 or new_x > x2 or new_y < y1 or new_y > y2:
            done = True 
            reward = 0
            info = 3
        
        # check collision with other agents
        if self.env[new_y, new_x] == CONSTANTS[f'p{opp_player}'] and not done:
            done = True
            reward = -10
            info = 2
        
        # check if ate food
        if self.env[new_y, new_x] == CONSTANTS['food']:
            p.grow()
            self.env[new_y, new_x] = CONSTANTS['background']
            self.food_x, self.food_y = self.get_randoms()
            self.env[self.food_y, self.food_x] = CONSTANTS['food']
            reward = 10
            info = 1
            

        if not done:
            # p.grow()
            pass
        else:
            # Add something to info # player is dead, displaye a cross or red patch
            pass 
        
        p.update(x, y)

        self.env[np.where(self.env == CONSTANTS[f'p{player}_head'])] = 0
        self.env[np.where(self.env == CONSTANTS[f'p{player}'])] = 0
        self.env = self.draw_boundary(self.env)       # redraw the boundaries
        self.env = p.draw(self.env, player=player)    # redraw the player
        #print(self.env)

        vision = p.look(self.env, self.agent_vision)  # Check if next state is required in case of done
        return np.hstack((vision.ravel(), (new_x - self.food_x, new_y - self.food_y))), reward, done, vision, info
