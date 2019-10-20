import numpy as np 
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

#import tensorflow as tf
#tf.get_logger().setLevel('INFO')

class QNetwork:

    def __init__(self,input_shape, hidden_units, output_shape, learning_rate=0.01):
        self.input_shape = input_shape
        self.model = Sequential()
     
        self.model.add(InputLayer((input_shape,)))
        #self.model.add(Flatten())
        for h in hidden_units:
            self.model.add(Dense(units=h, activation='relu'))
        self.model.add(Dense(units=output_shape, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate), loss='mse',metrics=['accuracy'])
        
    def predict(self, state, batch_size=1):
        return self.model.predict(state, batch_size)
    
    def train(self, states, action_values, batch_size, tensorboard_callback):
        self.model.fit(states, action_values, batch_size=batch_size, verbose=0, epochs=1, callbacks=[tensorboard_callback])


"""
import torch
import torch.nn as nn
import torch.functional as F

class QNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units):
        self.layer1 = nn.Linear(input_shape, hidden_units[0])
        self.layer2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.layer3 = nn.Linear(hidden_units[1], output_shape)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(x))
        out = F.relu(self.layer3(x))
        return out
"""