#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Zhizuo Chen (aka George)
# 
# Permission is hereby granted, free of charge, to use, copy, modify, and distribute this code for any purpose, provided that this notice is included in all copies or substantial portions of the code.
# 
# This code is provided "as is", without warranty of any kind, express or implied. The author shall not be liable for any damages or consequences arising from its use.
# 
# This file is associated with my arXiv preprint: https://arxiv.org/abs/2511.17598
# 
# For questions or bug reports, contact: zhizuo.chen@outlook.com

# In[ ]:


import numpy as np
import pandas as pd
import time, math, random, scipy
from copy import deepcopy
from collections import deque


# In[ ]:


#Basic settings.
global DFLT_GAMMA

#The default γₜ₊₁(s, a, s') value when no custom discount rate generating function is specified in the Gym Wrapper.
DFLT_GAMMA = .999


# In[ ]:





# # Wrappers

# In[ ]:


#Wrapper for the gymnasium environments.
class GymWrapper:

    #The constructor.
    def __init__(self, gym_envt):

        #The pointer to the gymnasium environment.
        self.gym_envt = gym_envt
        #func-end

    #Get the value of γₜ₊₁(s, a, s') (this function can be overloaded in the child class).
    def get_gamma(self, t, sta, act, n_sta): return DFLT_GAMMA
    #The function that returns a start time & state.
    def mu(self): return (0, (self.gym_envt.reset())[0])
    #The step function that returns the next state, the reward, and the termination status.
    def step(self, t, sta, act): return self.gym_envt.step(act)[: 3]
    #class-end


# In[ ]:


#A class for random selection among discrete actions.
class RAND_POL:

    #The constructor.
    def __init__(self, actions): self.__actions = actions
    #Select one action randomly.
    def select(self, *args): return random.choice(self.__actions)
    #class-end


# In[ ]:





# # Reward Noise Generators

# In[ ]:


#The function that adds normal noise (with zero expectation) to the costs.
def norm_noi_f(std): return std * np.random.randn()


# In[ ]:





# # Environments for Tabular Solutions

# ## The Tricky Gridworld Environment

# In[ ]:


#The class of the Trick Gridworld environment.
class TrickyGWD:

    #The constructor.
    def __init__(self, args_d, cost_noi_f):

        #Load gridworld length, width, windy zone coordinate, beginning & target locations, step costs and wind function.
        for k, v in args_d.items(): setattr(self, k, v)
        #Check whether the gridworld is long enough.
        assert self.length > self.windz + 2, "!!TrickyGridworld __init__ error: windy zone's length is at most 3!!"
        #The cost of taking one step in the Trick Gridworld.
        assert self.step_cost < 0, "!!TrickyGridworld __init__ error: costs (without noise) must be negative!!"
        #The function that adds noise to the costs (the noise has zero expectation).
        self.cost_noi_f = cost_noi_f
        #func-end
    
    #All possible states for the windy gridworld.
    def states(self): return tuple([(i + 1, j + 1) for i, j in np.ndindex(self.length, self.width)])
    #All available actions for the windy gridworld.
    def actions(self): return ('l', 'r', 'u', 'd')
    #The default start time & start location.
    def mu(self): return (0, self.beg)
    #The default function to get γₜ₊₁(s, a, s') (can be overloaded in the child class).
    def get_gamma(self, t, sta, act, n_sta): return DFLT_GAMMA
    
    #Take one step forward in one episode.
    def step(self, t, sta, act):
    
        #The episode is already terminated if {sta} is the target cell.
        if self.tgt == sta: return (sta, 0., True)
        #Get the coordinates of the current state.
        (x, y) = sta
        #Initialize the coordinates of the next state.
        nx, ny = x, y

        #The distance between the car and the windy zone.
        dist = x - self.windz
        #If the car is in the windy zone:
        if (dist >= 0) and (dist <= 2):
            #The strength of the winds.
            wind_str = self.wind_f(t)
            #The move caused by the wind.
            nx -= wind_str[dist]
            #if-end
        #The move caused by the action.
        if 'r' == act: nx += 1
        elif 'l' == act: nx -= 1
        elif 'u' == act: ny += 1
        else: ny -= 1
        
        #Get the next state.
        nx = int(np.clip(nx, 1, self.length))
        ny = int(np.clip(ny, 1, self.width))
        n_sta = (nx, ny)
        #Check whether the current episode is terminated & return:
        return (n_sta, self.step_cost + self.cost_noi_f(), True if self.tgt == n_sta else False)
        #func-end

    #The dynamics of the tricky gridworld problem.
    def dynamics(self, t, sta, act):
        
        #The dynamics can be deduced from the step function directly since the transitions are deterministic.
        n_sta = self.step(t, sta, act)[0]
        #Return the probabilities, next states, and rewards.
        return ([1.], [n_sta], [0. if self.tgt == sta else self.step_cost])
        #func-end
    #class-end


# In[ ]:


#The Tricky Gridworld with punish zones.
class Pun_TrickyGWD(TrickyGWD):

    #The constructor.
    def __init__(self, pun_hrzn, pun_locs, pun_gamma, args_d, cost_noi_f):
        
        #Call the constructor of the parent class.
        super().__init__(args_d, cost_noi_f)
        #Record the punishment time upper bound, the punishment locations, and the punishment discount rate.
        self.pun_hrzn, self.pun_locs, self.pun_gamma = pun_hrzn, pun_locs, pun_gamma
        #func-end
    
    #The function to get γₜ₊₁(s, a, s').
    def get_gamma(self, t, sta, act, n_sta):
        
        #Return punishment discount rates if reaching the punishment zone before passing the punshiment horizon.
        if (t < self.pun_hrzn) and (n_sta in self.pun_locs): return self.pun_gamma
        #Otherwise return the default discount rate.
        else: return DFLT_GAMMA
        #func-end
    #class-end


# In[ ]:





# In[ ]:




