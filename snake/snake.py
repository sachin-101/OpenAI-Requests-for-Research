from collections import deque
import numpy as np 
import time
import random

#colours - b, g, r
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
YELLOW = (255,255,0)

CONSTANTS = {'boundary':-5, 'p1':1, 'p1_head':2, 'background':0, 'p2':-1, 'p2_head':-2, 'food':5}

class Snake:

    def __init__(self, x_start, y_start, env_width, env_height):

        self.random_init(x_start, y_start)
        self.is_alive = True 
        self.moves = max(100, env_width*env_height*2)
        self.length = 4

#-----------Returns Random Start positions------------#

    def random_init(self, x_start, y_start):
        a = random.randint(-1,1)
        b = random.choice([1,-1]) if a == 0 else 0
        
        self.X = deque([x_start, x_start + a, x_start + 2*a, x_start + 3*a])
        self.Y = deque([y_start, y_start + b, y_start + 2*b, y_start + 3*b])

#-------------Draw the snake on the matrix--------------#
    
    def draw(self, env, player):

        for body_y, body_x in zip(self.Y, self.X):
            env[body_y, body_x] = CONSTANTS[f'p{player}']

        head_x, head_y = self.head_pos()    # head diff color
        env[head_y, head_x] = CONSTANTS[f'p{player}_head']
        return env 
            
#------------snake eat's food--------------#
    
    def grow(self):
        tail_x, tail_y = self.tail_pos()
        self.X.append(tail_x) 
        self.Y.append(tail_y)
        self.length += 1
        
#-----------move the snake forward-----------#
        
    def update(self, x_change, y_change):
        # updating snake's body position
        for i in range(self.length-1,0,-1):
            self.X[i] = self.X[i-1]
            self.Y[i] = self.Y[i-1]

        #updating head position
        self.X[0] = self.X[0] + x_change
        self.Y[0] = self.Y[0] + y_change 

#----------Observe the env and collect current state---------#

    def look(self, env, d):
        head_x, head_y = self.head_pos()
        view = env[head_y-d:head_y+d, head_x-d:head_x+d]
        return view

#------some helpful methods-----#

    def head_pos(self):
        return self.X[0], self.Y[0]
    
    def tail_pos(self):
        return self.X[-1], self.Y[-1]

    def kill(self):
        self.is_alive = False
    
    def set_length(self, length):
        self.length = length
    
