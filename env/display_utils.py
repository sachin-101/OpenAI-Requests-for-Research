'''
 Collection of functions which help to show hyperparameters and other important info
'''

import numpy as np
from cv2 import putText, FONT_HERSHEY_COMPLEX
from PIL import Image


# colors - b, g, r

#colours - b, g, r
WHITE = (255,255,255)
BLUE = (255,0,0)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

# text properties
font = FONT_HERSHEY_COMPLEX
extra_cols = 20

def add_states(env, state, action, left=False):
    rows, cols, channels = env.shape
    display_matrix = np.ones((rows, cols+extra_cols, channels), dtype=np.uint8)*200   # background
    if left:
        display_matrix[:rows, extra_cols:cols+extra_cols, :] = env 
    else:    
        display_matrix[:rows, 1:cols+1, :] = env 
    display_matrix = display_action(display_matrix, action, rows, left)
    display_matrix = display_state(display_matrix, state, rows, left)
    return display_matrix

def add_info(width, height, disp_matrix, p1_params, p2_params, train_params=None):
    '''
        Damn helpful function to show 
        all the stuff going on while Training
    '''
    rows, cols, channels = disp_matrix.shape
    img = Image.fromarray(disp_matrix, 'RGB')
    scale = int(width/cols)
    width = int(width + extra_cols*scale*2) + 400
    img = np.array(img.resize((width, height)))
    
    pad = 30
    x_pos1, x_pos2 = pad, width- pad - 150
    
    if train_params is not None:
        episode, eps, gamma = train_params.values()
        putText(img, 'Epsilon : {:.3f}'.format(eps), (int(width/2) - 200 , height - 10), font , 0.6, BLUE, 1)
        putText(img, f'Episode {episode}', (int(width/2), height - 20), font , 0.8, BLUE, 2)
        putText(img, f'Gamma : {gamma}', (int(width/2) + 200, height-10), font , 0.6, BLUE, 1)
    
    # Player 1
    putText(img, f'PLAYER {1}', (x_pos1, 70), font , 0.6, RED  , 1)
    stats_1 = p1_params['stats']
    putText(img, f'Food: {stats_1[1]}', (x_pos1, 100), font , 0.6, RED  , 1)
    putText(img, f'Hit {stats_1[2]}', (x_pos1, 130), font , 0.6, RED  , 1)
    putText(img, f'Boundary {stats_1[3]}', (x_pos1, 160), font , 0.6, RED  , 1)

    # Player 2
    putText(img, f'PLAYER {2}', (x_pos2, 70), font , 0.6, RED  , 1)
    stats_2 = p2_params['stats']
    putText(img, f'Food: {stats_2[1]}', (x_pos2, 100), font , 0.6, RED  , 1)
    putText(img, f'Hit {stats_2[2]}', (x_pos2, 130), font , 0.6, RED  , 1)
    putText(img, f'Boundary {stats_2[3]}', (x_pos2, 160), font , 0.6, RED  , 1)


    return img

def display_action(matrix, action, max_height, left):
    '''
        Helpful function to display the keys
        which display the action taken by the agent
    '''
    if left:
        x = 1
    else :
        x = -1    

    y = max_height - 5
    matrix[y, x*7,:] = BLUE   #left button
    matrix[y, x*9,:] = BLUE   #right button
    matrix[y-1, x*8,:] = BLUE  #up button
    matrix[y+1, x*8,:] = BLUE  #down button
    
    if action == 0:
        matrix[y, x*7,:] = RED   #left 
    elif action == 1:
        matrix[y, x*9,:] = RED   #right
    elif action == 2:
        matrix[y-1, x*8,:] = RED  #up 
    elif action ==3:
        matrix[y+1, x*8,:] = RED  #down 

    return matrix


def display_state(matrix, state, max_height, left):
    if left:
        x = 1
    else :
        x = -1    
    pad = 2
    mid_height = int(max_height/2) + 5
    h, w, c = state.shape
    if left:
        matrix[mid_height-10:mid_height-10+h,x*pad:x*pad+x*w,:] = state
    else:
        matrix[mid_height-10:mid_height-10+h,x*pad+x*w:x*pad,:] = state
    return matrix