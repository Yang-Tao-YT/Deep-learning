# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:58:20 2019

@author: yang
"""
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop,Adam
from keras.initializers import VarianceScaling

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """ Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
                    https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))
# In[]
class DQNAgent:
    """ Stock Trading Bot """

    def __init__(self, state_size, No = None , pretrained=False,model_name=None):
        '''agent config'''
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=1000)
        self.first_iter = True

        '''model config'''
        self.model_name = model_name
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {'huber_loss': huber_loss}  # important for loading the model from memory
        self.optimizer = RMSprop(lr=self.learning_rate)
        self.initializer = VarianceScaling()

        '''load pretrained model'''
        if pretrained and No != None:
            self.model = self.load(No)
            print('load trained model ' + '{}_{}'.format(model_name,No))
           
        else:
            self.model = self._model()

    def _model(self):
        """	Creates the model. """
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.state_size, kernel_initializer=self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=64, kernel_initializer=self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=64, kernel_initializer=self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=24, kernel_initializer=self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=self.action_size,activation = 'linear', kernel_initializer=self.initializer))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """ Adds relevant data to memory. """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """ Take action from given possible set of actions. """
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        if self.first_iter:
            self.first_iter = False
            return 1
        options = self.model.predict(state)
        return np.argmax(options[0])

    def train_experience_replay(self, batch_size):
        """ Train on previous experiences in memory. """

        batch = random.sample(self.memory, batch_size) 
    
        states = np.array([i[0][0] for i in batch]) # obtain the states
        action = np.array([i[1] for i in batch])# obtain the actions
        reward = np.array([i[2] for i in batch])# obtain the rewards
        next_states = np.array([i[3][0] for i in batch])# obtain the nex_states
        done = np.array([i[4] for i in batch])# obtain the if done or not
# =============================================================================
#         print(next_states, next_states.shape)
# =============================================================================
        # Q(s,a) = r + y*(Q(s',a'))
        Q = reward + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        Q[done] = reward[done] # for the last reward at the last period
        # set the target_Q to train the 
# =============================================================================
#         print(states, states.shape)
# =============================================================================
        target_Q = self.model.predict(states)
        target_Q[range(batch_size) , action] = Q

        # train the model
        self.model.fit(states ,target_Q , batch_size =batch_size,  epochs=1, verbose=0 )
        history = self.model.fit(states ,target_Q , batch_size =batch_size,  epochs=1, verbose=0 )
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save('{}_{}.h5'.format(self.model_name, episode))

    def load(self,episode):
        return load_model('{}_{}.h5'.format(self.model_name, episode))