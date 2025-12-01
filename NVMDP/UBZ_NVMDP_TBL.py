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


import os, sys
#Get the parent directory of the current working directory.
prnt_dir = os.path.dirname(os.getcwd())
#Add more directories via relative paths for Python module loading.
sys.path.append(os.path.normpath(os.path.join(prnt_dir, 'DSA')))


# In[ ]:


import numpy as np
import pandas as pd
import time, math, random, scipy
from collections import defaultdict
import UBZ_DSA


# In[ ]:





# # Basic Tools

# In[ ]:


#Calculate the returns from gamma values and rewards.
def rwd_2_ret(gam_seq, rwd_seq, last_v):

    #Get the length of the γ value sequence.
    steps = len(gam_seq)
    #Check whether the reward seqeucne has the same number of elements.
    assert steps == len(rwd_seq), "!!rwd_2_ret error: the lengths of the gamma & the reward sequences are not equal!!"
    
    #Initialize the return with the (estimated) value of the last state.
    ret = last_v
    #The container of the returns.
    ret_seq = list()
    #Calculate the returns backwards through time steps.
    for i in range(steps - 1, -1, -1):
        #Calculate the return of time step {t}.
        ret = rwd_seq[i] + gam_seq[i] * ret
        #Store the return of time step {t} in the container.
        ret_seq.append(ret)
        #for-end
    #Reverse the order of the container then return:
    ret_seq.reverse()
    return ret_seq
    #func-end


# In[ ]:


#Calculate state-values from action-values and a given policy.
def av_2_sv(t0, te, states, av_res, pol):
    
    #Compare the lengths of action sets in {av_res} and {pol}.
    assert len(av_res.actions) == len(pol.actions), "!!av_2_sv error: Action-values & policy have different action sets!!"
    #The container of the state-values.
    sv_res = SValStruc()
    
    #Loop through all times.
    for t in range(t0, te):
        #Loop through all states.
        for sta in states:
            #Generate the time-state pair.
            t_sta = (t, sta)
            #Get the action-values & the sub-policy under {sta} at {t}.
            spol, avd = pol.get_sta_pol_(t_sta), av_res.get_sta_allav_(t_sta)
            #Calculate the current state value.
            sv = sum(spol[a] * avd[a] for a in pol.actions)
            #Update the state-values.
            sv_res.update_(t_sta, sv)
            #inner-for-end
        #for-end
    #Return the state-values:
    return sv_res
    #func-end


# In[ ]:


#Run the environment for one episode with a specified policy {pol}.
def run_envt(te, envt, pol):
    
    #The sequences of states, actions, discount rates, and rewards in the test episode.
    sta_seq, act_seq, gam_seq, rwd_seq = list(), list(), list(), list()
    #Initialize the start time and state.
    t0, sta = envt.mu()
    
    #Iteration through the current episode.
    for t in range(t0, te):
        #Choose one action according to the specified policy.
        act = pol.select(t, sta)
        #Take one step forward.
        (n_sta, rwd, ts) = envt.step(t, sta, act)
        #Record the current state, action, γₜ₊₁(s, a, s'), and rₜ(s, a, s').
        sta_seq.append(sta)
        act_seq.append(act)
        gam_seq.append(envt.get_gamma(t, sta, act, n_sta))
        rwd_seq.append(rwd)
        #Break the loop if the current episode is ended.
        if ts: break
        #The next state {n_sta} becomes the current state in the next step.
        sta = n_sta
        #inner-for-end
    #Record the time steps & termination status.
    tnts = (t0, t, ts)
    #Record the last time state.
    sta_seq.append(n_sta)
    #Return:
    return (tnts, sta_seq, act_seq, gam_seq, rwd_seq)
    #func-end


# In[ ]:


#The class that returns a constant.
class StepSize:
    
    #The constructor.
    def __init__(self, args, upd_f):
        
        #The steps taken.
        self.taken_steps = 0
        #Record the arguments & update function.
        self.args = args
        self.upd_f = upd_f
        #func-end
    
    #Overload the "()" operation.
    def __call__(self):

        #Calculate the current step size.
        ss = self.upd_f(self.taken_steps, self.args)
        #Increase the steps taken.
        self.taken_steps += 1
        #Return the new step size.
        return ss
        #func-end
    #class-end


# In[ ]:





# # The NVMDP state-value, action-value, and policy structures

# ## The state-value structure

# In[ ]:


#The structure for storing state-values in finite state-space.
#The default value for unseen time-state pair is float-zero.
class SValStruc(defaultdict):

    #The constructor.
    def __init__(self): super().__init__(float)
    
    #If other default values are needed, inherit this class and override this "__missing__" function.
    """def __missing__(self, t_sta): return random.gauss()"""
    #Get a specified state-value.
    def get_(self, t_sta): return self[t_sta]
    #Update the state-value of a specified state.
    def update_(self, t_sta, n_sv): self[t_sta] = n_sv
    #class-end


# ## The action-value structure

# In[ ]:


#The structure for storing action-values in finite state- & action-space.
#!!Notice: The following implementation is not optimized for Q-learning!!
class AValStruc(dict):

    #The constructor.
    def __init__(self, actions):
        
        #Record all available actions.
        self.actions = actions if type(actions) is tuple else tuple(actions)
        #Record the number of available actions.
        self.actnum = len(actions)
        #Record the indexes for all available actions.
        self.act2ind = dict()
        for ind, act in enumerate(self.actions): self.act2ind[act] = ind
        #func-end
    
    #If other default values are needed, inherit this class and override this "__missing__" function.
    def __missing__(self, t_sta):
        
        #Set the entry in the dict & return.
        v = [0.] * self.actnum
        self[t_sta] = v
        return v
        #func-end
    
    #Get a specified action-value under {sta} at {t}.
    def get_(self, t_sta, act): return (self[t_sta])[self.act2ind[act]]
    #Update the action-value of a specified state-action pair at {t}.
    def update_(self, t_sta, act, n_av): (self[t_sta])[self.act2ind[act]] = n_av
    
    #Get all action-values under {sta} at {t}.
    def get_sta_allav_(self, t_sta):
        
        #Get all action-values under {sta} at {t}.
        allav = self[t_sta]
        #Return the dictionary of all action-values under {sta} at {t}.
        return {a: allav[self.act2ind[a]] for a in self.actions}
    
    #Get the action with the maximum action-value under {sta} at {t}.
    def get_q_act_(self, t_sta):
        
        #Get all action-values under {sta} at {t}.
        allav = self[t_sta]
        #Find the action index that corresponds to the maximum action-value.
        m_ind = np.argmax(allav)
        #Return the action with maximum action-value & its correpsonding action-value.
        return (self.actions[m_ind], allav[m_ind])
        #func-end
    
    #Select an action ε-greedily.
    def eps_act_sel_(self, t_sta, eps):
        
        #Choose one action derived from the action-values in an ε-greedy manner.
        if random.random() < eps: return random.choice(self.actions)
        else: return (self.get_q_act_(t_sta))[0]
        #func-end
    
    #Generate an ε-greedy policy from the action-values.
    def av2pol(self, eps):
        
        #Initialize a policy object.
        eps_pol = PolStruc(self.actions)
        #Loop through all time-state pairs stored in {self.a_vals}.
        for t_sta, allav in self.items():
            #Find the action index that corresponds to the maximum action-value.
            m_ind = np.argmax(allav)
            #Update the sub-policy under state {sta} at {t}.
            spol = [eps / self.actnum] * self.actnum
            spol[m_ind] += 1. - eps
            eps_pol[t_sta] = spol
            #for-end
        #Return the ε-greedy policy.
        return eps_pol
        #func-end
    #class-end


# ## The policy structure

# In[ ]:


#The structure for storing policies in finite state- & action-space.
class PolStruc(dict):

    #The constructor.
    def __init__(self, actions):
        
        #Record the available actions.
        self.actions = actions if type(actions) is tuple else tuple(actions)
        #Record the number of available actions.
        self.actnum = len(actions)
        #Record the indexes for all available actions.
        self.act2ind = dict()
        for ind, act in enumerate(self.actions): self.act2ind[act] = ind
        #func-end

    #If other default values are needed, inherit this class and override this "__missing__" function.
    def __missing__(self, t_sta):
        
        #Set the entry in the dict & return.
        v = [1. / self.actnum] * self.actnum
        self[t_sta] = v
        return v
        #func-end
    
    #Get the probability of selecting {act} under {sta} at {t}.
    def get_(self, t_sta, act): return (self[t_sta])[self.act2ind[act]]
    #Select an action under {sta} at {t}.
    def select_(self, t_sta): return (random.choices(self.actions, weights = self[t_sta]))[0]
    #Select an action under {sta} at {t} (times & states are separate).
    def select(self, t, sta): return (random.choices(self.actions, weights = self[(t, sta)]))[0]
    #Get the sub-policy in a dict under {sta} at time {t}
    def get_sta_pol_(self, t_sta): return {a: pr for a, pr in zip(self.actions, self[t_sta])}
    
    #Get the action with the most likelihood (max probability) under {sta} at {t}.
    def get_ml_act_(self, t_sta):
        
        #Get the probabilities under {sta} at {t}.
        spol = self[t_sta]
        #Return the action with the max probability.
        ml_act_ind = np.argmax(spol)
        return (self.actions[ml_act_ind], spol[ml_act_ind])
        #func-end
    
    #Update the probabilities of several actions under {sta} at {t}.
    #Notice: {d_probs} should be a dict: {act: prob}, and all its action keys are subset of the action space.
    def update_(self, t_sta, d_probs):
        
        #The count of actions that do not appear in {d_probs} must be nonnegative.
        ct_diff = self.actnum - len(d_probs)
        assert ct_diff >= 0, "!!PolStruc.update error: d_probs has too many actions!!"
        #Get the sum of probabilities in {d_probs}.
        pr_sum = 0.
        for pr in d_probs.values(): pr_sum += pr
        #Calculate the default probability of each action that does not appear in {d_probs}.
        assert (pr_sum >= 0.) and (pr_sum <= 1.), "!!PolStruc.update error: d_probs has invalid probabilites!!"
        dflt_pr = 0. if 0 == ct_diff else (1. - pr_sum) / ct_diff
        #Get the probabilities under {sta} at {t}.
        spol = self[t_sta]
        #Update all the probabilities.
        for ind, a in enumerate(self.actions):
            #Get the new action value, otherwise fill in the default probability.
            n_pr = d_probs.get(a)
            if n_pr is None: spol[ind] = dflt_pr
            else: spol[ind] = n_pr
            #for-end
        #func-end
    #class-end


# In[ ]:





# # Generalized Q-learning

# ## The action-value structure for Generalized Q-learning

# In[ ]:


#The action-value estimation structure for one time-state pair in Generalized Q-learning.
class GeQ_TSUnit:

    #The constructor.
    def __init__(self, actions, est_num, hist_num, same_f_flag):
        
        #The container of the action-value estimates under the current time-state pair.
        self.av_ests = dict()
        #If {same_f_flag} is True (i.e., f₁ = f₂ = ... = fₙ in Generalized Q-learning Algorithm).
        if same_f_flag:
            #Only one RB-tree will be assigned to {self.av_reps}.
            rbt = UBZ_DSA.RBTree()
            self.av_reps = tuple(rbt for _ in range(est_num))
            #Loop through all actions:
            for a in actions:
                #The 2-D size matix to store all action-value estimates (each row stands for one estimate track).
                ests = np.zeros([est_num, hist_num], dtype = np.float64)
                #The column indices for next estimates on each row (i.e., estimate track).
                nx_cols = [0] * est_num
                #Insert a new node into the RB-tree of representative values & set the node's action attribute.
                node = self.av_reps[0].rb_insert(0.)
                setattr(node, 'act', a)
                #Store the current estimate tracks, column indices, and the node of the representative value in the dict.
                self.av_ests[a] = [ests, nx_cols, node]
                #for-end
        #Otherwise {same_f_flag} is False (i.e., f₁, f₂, ..., fₙ are different in Generalized Q-learning Algorithm).
        else:
            #{est_num} RB-trees will be contained in {self.av_reps}.
            self.av_reps = tuple(UBZ_DSA.RBTree() for _ in range(est_num))
            #Loop through all actions:
            for a in actions:
                #The 2-D size matix to store all action-value estimates (each row stands for one estimate track).
                ests = np.zeros([est_num, hist_num], dtype = np.float64)
                #The column indices for next estimates on each row (i.e., estimate track).
                nx_cols = [0] * est_num
                #The container of representative values of action {a} across all estimate tracks.
                nodes = list()
                #Insert a new node into each RB-tree of representative values & set the node's action attribute.
                for i in range(est_num):
                    node = self.av_reps[i].rb_insert(0.)
                    setattr(node, 'act', a)
                    nodes.append(node)
                    #inner-for-end
                #Store the current estimate tracks, column indices, and the node of the representative value in the dict.
                self.av_ests[a] = [ests, nx_cols, nodes]
                #for-end
            #if-end
        #func-end
    
    #Get the latest estimate of the {est-ind}-th estimate track for {act}.
    def get_est(self, act, est_ind):
        
        #Get the estimate tracks & update column indices of {act}.
        ests, nx_cols = (self.av_ests[act])[: 2]
        #Get & return the latest estimate on the {est-ind}-th estimate track (-1 is equivalent to the last element).
        return ests[est_ind, nx_cols[est_ind] - 1]
        #func-end
    #class-end


# In[ ]:


#The action-value structure for Maxmin Q-learning with Generalized Q-learning framework.
class Maxmin_AvalStruc(dict):

    #The constructor.
    def __init__(self, actions, est_num, hist_num):
        
        #Record all available actions.
        self.actions = actions if type(actions) is tuple else tuple(actions)
        #Record the number of available actions.
        self.actnum = len(actions)
        #The number of action-value estimates for each action, and the number of historical values for each estimate.
        self.est_num, self.hist_num = est_num, hist_num
        #func-end
    
    #The "__missing__(self, t_sta)" function designed for Maxmin Q-learning (f₁ = f₂ = ... = fₙ).
    def __missing__(self, t_sta):

        #Initialize an action-value estimation structure.
        #!!Notice: same_f_flag is True for Maxmin Q-learning!!
        tsunit = GeQ_TSUnit(self.actions, self.est_num, self.hist_num, True)
        #Set the entry in the dict & return.
        self[t_sta] = tsunit
        return tsunit
        #func-end
    
    #Get the latest estimate of the {est-ind}-th estimate track for {act} under state {sta} at time {t}.
    def get_est_(self, t_sta, act, est_ind): return (self[t_sta]).get_est(act, est_ind)
    
    #Store the latest estimate of the {est-ind}-th estimate track for {act} under state {sta} at time {t}.
    def update_est_(self, t_sta, act, est_ind, av_new):
        
        #Get the pointer to the time-state pair and {av_ests[act]} in it.
        tsunit = self[t_sta]
        av_ests_act = tsunit.av_ests[act]
        #Get the estimate tracks, update column indices, and the node of the representative value of action {act}.
        ests, nx_cols, node = av_ests_act[: 3]
        #Get the column to update in the {est-ind}-th estimate track.
        upd_col = nx_cols[est_ind]
        #Update the {est-ind}-th estimate track with the latest estimate value.
        av_old = ests[est_ind, upd_col]
        ests[est_ind, upd_col] = av_new
        #Update the column index for the next update in the {est-ind}-th estimate track.
        nx_cols[est_ind] = (upd_col + 1) % ests.shape[1]
        #If the new action-value estimate is smaller than the old representative, it becomes the new minimum.
        if av_new < node.val: rep_new = av_new
        #Otherwise, if the old action-value is not the smallest, then the old representative remains.
        elif av_old > node.val: rep_new = None
        #The new minimum needs to be re-calculated since the old action-value is the old representative.
        else: rep_new = ests.min()
        #Update the container (RB-tree) of all representative action-values.
        if rep_new is not None:
            av_rep_ptr = tsunit.av_reps[0]
            av_rep_ptr.rb_delete_ptr(node)
            new_node = av_rep_ptr.rb_insert(rep_new)
            setattr(new_node, 'act', act)
            #Store the new node in {self.av_ests[act]}.
            av_ests_act[2] = new_node
            #if-end
        #func-end
    
    #Get the action with the maximum representative action-value under {sta} at {t}.
    def get_q_act_(self, t_sta, est_ind):
        
        #Find the node of maximum representative action-value under {sta} at {t}.
        max_node = (self[t_sta].av_reps[est_ind]).find_max()
        #Return the action with maximum representative action-value & its correpsonding representative action-value.
        return (max_node.act, max_node.val)
        #func-end
    
    #Select an action ε-greedily.
    def eps_act_sel_(self, t_sta, est_ind, eps):
        
        #Choose one action derived from the action-values in an ε-greedy manner.
        if random.random() < eps: return random.choice(self.actions)
        else: return (self.get_q_act_(t_sta, est_ind))[0]
        #func-end
    
    #Generate an ε-greedy policy from the action-values.
    def av2pol(self, eps):
        
        #Initialize a policy object.
        eps_pol = PolStruc(self.actions)
        #Loop through all time-state pairs stored in {self.a_vals}.
        for t_sta, tsunit in self.items():
            #Find the node of maximum action-value.
            max_node = tsunit.av_reps[0].find_max()
            #Update the sub-policy under state {sta} at {t}.
            spol = [eps / self.actnum] * self.actnum
            spol[eps_pol.act2ind[max_node.act]] += 1. - eps
            eps_pol[t_sta] = spol
            #for-end
        #Return the ε-greedy policy.
        return eps_pol
        #func-end
    #class-end


# In[ ]:


#The action-value structure for Average Q-learning with Generalized Q-learning framework.
class Avg_AvalStruc(Maxmin_AvalStruc):

    #The constructor.
    def __init__(self, actions, est_num, hist_num):

        #Call the constructor of the parent class.
        super().__init__(actions, est_num, hist_num)
        #Record the total number of elements in the estimation matrix of each time-state unit.
        self.mat_size = est_num * hist_num
        #func-end
    
    #Store the latest estimate of the {est-ind}-th estimate track for {act} under state {sta} at time {t}.
    def update_est_(self, t_sta, act, est_ind, av_new):
        
        #Get the pointer to the time-state pair and {av_ests[act]} in it.
        tsunit = self[t_sta]
        av_ests_act = tsunit.av_ests[act]
        #Get the estimate tracks, update column indices, and the node of the representative value of action {act}.
        ests, nx_cols, node = av_ests_act[: 3]
        #Get the column to update in the {est-ind}-th estimate track.
        upd_col = nx_cols[est_ind]
        #Update the {est-ind}-th estimate track with the latest estimate value.
        av_old = ests[est_ind, upd_col]
        ests[est_ind, upd_col] = av_new
        #Update the column index for the next update in the {est-ind}-th estimate track.
        nx_cols[est_ind] = (upd_col + 1) % ests.shape[1]
        #Calculate the new representative value.
        rep_new = node.val + (av_new - av_old) / self.mat_size
        #Update the container (RB-tree) of all representative action-values.
        av_rep_ptr = tsunit.av_reps[0]
        av_rep_ptr.rb_delete_ptr(node)
        new_node = av_rep_ptr.rb_insert(rep_new)
        setattr(new_node, 'act', act)
        #Store the new node in {self.av_ests[act]}.
        av_ests_act[2] = new_node
        #func-end
    #class-end


# In[ ]:


#The action-value structure for Present-Track-Maxmin Q-learning with Generalized Q-learning framework.
class PTMaxmin_AvalStruc(Maxmin_AvalStruc):

    #Overload the "__missing__(self, t_sta)" function from Maxmin Q-learning.
    #!!Notice: same_f_flag is False for Present-Track-Maxmin Q-learning!!
    def __missing__(self, t_sta):
        
        #Initialize an action-value estimation structure.
        tsunit = GeQ_TSUnit(self.actions, self.est_num, self.hist_num, False)
        #Set the entry in the dict & return.
        self[t_sta] = tsunit
        return tsunit
        #func-end
    
    #Store the latest estimate of the {est-ind}-th estimate track for {act} under state {sta} at time {t}.
    def update_est_(self, t_sta, act, est_ind, av_new):
        
        #Get the pointer to the time-state pair and {av_ests[act]} in it.
        tsunit = self[t_sta]
        av_ests_act = tsunit.av_ests[act]
        #Get the estimate tracks, update column indices, and the node of the representative value of action {act}.
        ests, nx_cols, nodes = av_ests_act[: 3]
        #Get the column to update in the {est-ind}-th estimate track.
        upd_col = nx_cols[est_ind]
        #Update the {est-ind}-th estimate track with the latest estimate value.
        av_old = ests[est_ind, upd_col]
        ests[est_ind, upd_col] = av_new
        #Update the column index for the next update in the {est-ind}-th estimate track.
        nx_cols[est_ind] = (upd_col + 1) % ests.shape[1]
        #Get the old representative value in the {est-ind}-th estimate track.
        rep_old = nodes[est_ind].val
        #If the new action-value estimate is smaller than the old representative, it becomes the new minimum.
        if av_new < rep_old: rep_new = av_new
        #Otherwise, if the old action-value is not the smallest, then the old representative remains.
        elif av_old > rep_old: rep_new = None
        #The new minimum needs to be re-calculated since the old action-value is the old representative.
        else: rep_new = ests[est_ind, :].min()
        #Update the container (RB-tree) of representative action-values in the {est-ind}-th estimate track.
        if rep_new is not None:
            av_rep_ptr = tsunit.av_reps[est_ind]
            av_rep_ptr.rb_delete_ptr(nodes[est_ind])
            new_node = av_rep_ptr.rb_insert(rep_new)
            setattr(new_node, 'act', act)
            #Store the new node in {self.av_ests[act]}.
            (av_ests_act[2])[est_ind] = new_node
            #if-end
        #func-end
    #class-end


# In[ ]:


#The action-value structure for Weighted-Average Q-learning with Generalized Q-learning framework.
class WtAvg_AvalStruc(Maxmin_AvalStruc):

    #The constructor.
    def __init__(self, actions, est_num, hist_num, la, eta):
        
        #Check whether λ is positive and not equal to 1.
        assert (la > 0.) and (la != 1.), "!!WtAvg_AvalStruc.__init__ error: λ must be positive and not 1!!"
        #Check whether η is within [0, 1].
        assert (eta >= 0.) and (eta <= 1.), "!!WtAvg_AvalStruc.__init__ error: η must be within [0, 1]!!"
        #Call the constructor of the parent class.
        super().__init__(actions, est_num, hist_num)
        #Record λ.
        self.la = la
        #Calculate & store the first weight (w₀ = 1 / (1 + λ + λ² + ... + λˡ⁻¹)), and the last one (wₗ₋₁ = λˡ⁻¹ w₀).
        self.w0 = (1. - la) / (1. - la ** hist_num)
        self.we = (la ** (hist_num - 1)) * self.w0
        #Record η & (1 - η) / (1 - {est_num}).
        if 1 == est_num: self.eta, self.onesubeta_split = 1., 0.
        else: self.eta, self.onesubeta_split = eta, (1. - eta) / (est_num - 1)
        #func-end

    #Overload the "__missing__(self, t_sta)" function from Maxmin Q-learning.
    #!!Notice: same_f_flag is False for Weighted-Average Q-learning!!
    def __missing__(self, t_sta):
        
        #Initialize an action-value estimation structure.
        tsunit = GeQ_TSUnit(self.actions, self.est_num, self.hist_num, False)
        #Store the weighted row sum.
        for a in self.actions: (tsunit.av_ests[a]).append([0.] * self.est_num)
        #Set the entry in the dict & return.
        self[t_sta] = tsunit
        return tsunit
        #func-end
    
    #Store the latest estimate of the {est-ind}-th estimate track for {act} under state {sta} at time {t}.
    def update_est_(self, t_sta, act, est_ind, av_new):
        
        #Get the pointer to the time-state pair and {av_ests[act]} in it.
        tsunit = self[t_sta]
        av_ests_act = tsunit.av_ests[act]
        #Get the estimate tracks, update column indices, representative values of action {act}, and the weighted row sums.
        ests, nx_cols, nodes, row_wsums = av_ests_act
        #Get the column to update in the {est-ind}-th estimate track.
        upd_col = nx_cols[est_ind]
        #Update the {est-ind}-th estimate track with the latest estimate value.
        av_old = ests[est_ind, upd_col]
        ests[est_ind, upd_col] = av_new
        #Update the column index for the next update in the {est-ind}-th estimate track.
        nx_cols[est_ind] = (upd_col + 1) % ests.shape[1]
        #Get the old weighted row sum in the {est-ind}-th estimate track.
        old_wsum = row_wsums[est_ind]
        #Calculate & update the new weighted row sum in the {est-ind}-th estimate track.
        new_wsum = self.w0 * av_new + self.la * (old_wsum - self.we * av_old)
        row_wsums[est_ind] = new_wsum
        #The container of representative values of action {a} across all estimate tracks.
        l_new_nodes = list()
        #Update the container (RB-tree) of all representative action-values.
        for i in range(self.est_num):
            #If {i} is the current estimate track:
            if est_ind == i:
                #Calculate the new representative value in the {est-ind}-th estimate track.
                rep_new = nodes[est_ind].val + self.eta * (new_wsum - old_wsum)
                #Update the container (RB-tree) of representative action-values in the {est-ind}-th estimate track.
                av_rep_ptr = tsunit.av_reps[est_ind]
                av_rep_ptr.rb_delete_ptr(nodes[est_ind])
                new_node = av_rep_ptr.rb_insert(rep_new)
                setattr(new_node, 'act', act)
                #Add the new node to the new container.
                l_new_nodes.append(new_node)
            #Otherwise:
            else:
                #Calculate the new representative value in the {est-ind}-th estimate track.
                rep_new = nodes[i].val + self.onesubeta_split * (new_wsum - old_wsum)
                #Update the container (RB-tree) of representative action-values in the {i}-th estimate track.
                av_rep_ptr = tsunit.av_reps[i]
                av_rep_ptr.rb_delete_ptr(nodes[i])
                new_node = av_rep_ptr.rb_insert(rep_new)
                setattr(new_node, 'act', act)
                #Add the new node to the new container.
                l_new_nodes.append(new_node)
                #if-end
            #for-end
        #Store the new nodes in {self.av_ests[act]}.
        av_ests_act[2] = l_new_nodes
        #func-end
    #class-end


# In[ ]:





# # Dynamic Programming for NVMDPs

# In[ ]:


#DP policy evaluation for NVMDP.
def dp_pol_eval(t0, te, envt, pol, sv_res, av_res):

    #Get all states & actions from the finite MDP.
    states, actions = envt.states(), envt.actions()
    
    #Calculate the state-values & action-values backwards.
    for t in range(te - 1, t0 - 1, -1):
        #Loop through all states:
        for sta in states:
            #Generate the time-state pair & calculate t plus 1.
            t_sta = (t, sta)
            t_plus_1 = t + 1
            #Initiliaze the current state value to zero.
            sv = 0.
            #Loop through all actions:
            for act in actions:
                #Get the probabilities of all possible next states & the corresponding mean rewards.
                (prob_seq, n_sta_seq, rwd_seq) = envt.dynamics(t, sta, act)
                #Calculate the action-value of the current state-action pair at {t}.
                av = 0.
                for (pr, n_sta, rwd) in zip(prob_seq, n_sta_seq, rwd_seq):
                    av += pr * (rwd + envt.get_gamma(t, sta, act, n_sta) * sv_res.get_((t_plus_1, n_sta)))
                    #innermost-for-end
                #Update the container of action-values.
                av_res.update_(t_sta, act, av)
                #Update the current state-value.
                sv += pol.get_(t_sta, act) * av
                #inner^2-for-end
            #Update the container of state-values.
            sv_res.update_(t_sta, sv)
            #inner-for-end
        #for-end
    #func-end


# In[ ]:


#DP value iteration for NVMDP.
def dp_val_iter(t0, te, envt, sv_res, av_res):

    #Get all states & actions from the finite MDP.
    states, actions = envt.states(), envt.actions()
    #Initialize the optimal policy derived from the learned action-values.
    q_pol = PolStruc(envt.actions())

    #Calculate the state-values & action-values backward.
    for t in range(te - 1, t0 - 1, -1):
        #Loop through all states:
        for sta in states:
            #Generate the time-state pair & calculate t plus 1.
            t_sta = (t, sta)
            t_plus_1 = t + 1
            #Initialize the optimal state value to negative infinity, and the optimal action to null.
            sv, q_act = -np.inf, None
            #Loop through all actions:
            for act in actions:
                #Get the probabilities of all possible next states & the corresponding mean rewards.
                (prob_seq, n_sta_seq, rwd_seq) = envt.dynamics(t, sta, act)
                #Calculate the action-value of the current state-action pair at {t}.
                av = 0.
                for (pr, n_sta, rwd) in zip(prob_seq, n_sta_seq, rwd_seq):
                    av += pr * (rwd + envt.get_gamma(t, sta, act, n_sta) * sv_res.get_((t_plus_1, n_sta)))
                    #innermost-for-end
                #Update the container of action-values.
                av_res.update_(t_sta, act, av)
                #Update the optimal state-value & action.
                if av > sv: sv, q_act = av, act
                #inner^2-for-end
            #Update the container of state-values.
            sv_res.update_(t_sta, sv)
            #Update the optimal policy.
            q_pol.update_(t_sta, {q_act: 1.})
            #inner-for-end
        #for-end
    #Return the learned optimal policy:
    return q_pol
    #func-end


# In[ ]:





# # Monte Carlo Methods for NVMDPs

# In[ ]:


#MC policy evaluation for NVMDP.
def mc_pol_eval(te, envt, pol, ep_num, sv_res, sv_ct_res, av_res, av_ct_res):
    
    #Run the environment for {ep_num} times.
    for ep in range(ep_num):

        #The sequences of states, actions, γ values, rewards, and returns in the test episode.
        sta_seq, act_seq, gam_seq, rwd_seq, ret_seq = list(), list(), list(), list(), list()
        #Initialize the start time & state.
        t0, sta = envt.mu()
        
        #Iteration through the current episode.
        for t in range(t0, te):
            #Choose one action according to the specified policy.
            act = pol.select(t, sta)
            #Take one step forward.
            (n_sta, rwd, ts) = envt.step(t, sta, act)
            #Record the current state, action, γₜ₊₁(s, a, s'), and rₜ(s, a, s').
            sta_seq.append(sta)
            act_seq.append(act)
            gam_seq.append(envt.get_gamma(t, sta, act, n_sta))
            rwd_seq.append(rwd)
            #Break the loop if the current episode is ended.
            if ts: break
            #The next state {n_sta} becomes the current state in the next step.
            sta = n_sta
            #inner-for-end
        #Calculate the returns of the test episode.
        ret_seq = rwd_2_ret(gam_seq, rwd_seq, last_v = 0. if ts else sv_res.get_((t + 1, n_sta)))

        #Update the Monte Carlo estimates of state- & action- values.
        for t, (sta, act, ret) in enumerate(zip(sta_seq, act_seq, ret_seq), start = t0):
            #Generate the time-state pair.
            t_sta = (t, sta)
            #Get & update the count of the current state-value.
            sv_ct = int(sv_ct_res.get_(t_sta)) + 1
            sv_ct_res.update_(t_sta, sv_ct)
            #Get & update the Monte Carlo estimate of the current state-value.
            sv = sv_res.get_(t_sta)
            sv_res.update_(t_sta, sv + (ret - sv) / sv_ct)
            #Get & update the count of the current action-value.
            av_ct = int(av_ct_res.get_(t_sta, act)) + 1
            av_ct_res.update_(t_sta, act, av_ct)
            #Get & update the Monte Carlo estimate of the current action-value.
            av = av_res.get_(t_sta, act)
            av_res.update_(t_sta, act, av + (ret - av) / av_ct)
            #inner-for-end
        #for-end
    #func-end


# In[ ]:


#On-policy MC control (with ε-soft policies) for NVMDP.
def mc_control_esoft(te, envt, pol, eps, ep_num, alpha, av_res):
    
    #Calculate the probability for the ε-greedy action.
    gd_act_pr = 1. - eps + eps / len(envt.actions())
    
    #Run the environment for {ep_num} times.
    for ep in range(ep_num):

        #The sequences of states, actions, γ values, rewards, and returns in the test episode.
        sta_seq, act_seq, gam_seq, rwd_seq, ret_seq = list(), list(), list(), list(), list()
        #Initialize the start time & state.
        t0, sta = envt.mu()
        
        #Iteration through the current episode.
        for t in range(t0, te):
            #Choose one action according to the specified policy.
            act = pol.select(t, sta)
            #Take one step forward.
            (n_sta, rwd, ts) = envt.step(t, sta, act)
            #Record the current state, action, γₜ₊₁(s, a, s'), and rₜ(s, a, s').
            sta_seq.append(sta)
            act_seq.append(act)
            gam_seq.append(envt.get_gamma(t, sta, act, n_sta))
            rwd_seq.append(rwd)
            #Break the loop if the current episode is ended.
            if ts: break
            #The next state {n_sta} becomes the current state in the next step.
            sta = n_sta
            #inner-for-end
        #Calculate the returns of the test episode.
        ret_seq = rwd_2_ret(gam_seq, rwd_seq, last_v = 0. if ts else (av_res.get_q_act_((t + 1, n_sta)))[1])

        #Update the Monte Carlo estimates of action-values & the policy.
        for t, (sta, act, ret) in enumerate(zip(sta_seq, act_seq, ret_seq), start = t0):
            #Generate the time-state pair.
            t_sta = (t, sta)
            #Get & update the Monte Carlo estimate of the current action-value.
            av = av_res.get_(t_sta, act)
            av_res.update_(t_sta, act, av + alpha() * (ret - av))
            #Find the ε-greedy action & update the policy.
            q_act = av_res.get_q_act_(t_sta)[0]
            pol.update_(t_sta, {q_act: gd_act_pr})
            #inner-for-end
        #for-end
    #func-end


# In[ ]:





# # Temporal-Difference Learning for NVMDPs

# In[ ]:


#Tabular TD(0) for policy evaluation.
def td0(te, envt, pol, ep_num, alpha, sv_res):
    
    #Run the environment for {ep_num} times.
    for ep in range(ep_num):
        
        #Initialize the start time & state.
        t0, n_sta = envt.mu()
        #Initialize the next time-state pair.
        nxt_t_sta = (t0, n_sta)
        #Iteration through the current episode.
        for t in range(t0, te):
            #The current state & time-state pair comes from the previous step or initialization.
            sta, t_sta = n_sta, nxt_t_sta
            #Choose one action according to the specified policy.
            act = pol.select_(t_sta)
            #Take one step forward.
            (n_sta, rwd, ts) = envt.step(t, sta, act)
            #Generate the next time-state pair.
            nxt_t_sta = (t + 1, n_sta)
            #Calculate the new state-value.
            sv, n_sv = sv_res.get_(t_sta), 0. if ts else sv_res.get_(nxt_t_sta)
            sv_new_val = sv + alpha() * (rwd + envt.get_gamma(t, sta, act, n_sta) * n_sv - sv)
            #Update the current state-value.
            sv_res.update_(t_sta, sv_new_val)
            #Break the loop if the current episode is ended.
            if ts: break
            #inner-for-end
        #for-end
    #func-end


# In[ ]:





# # Q-learning for NVMDPs

# ## NVMDP Q-learning

# In[ ]:


#Q-learning for policy improvement.
def q_learn(te, envt, sel_act_, ep_num, alpha, av_res):
    
    #Get all available actions.
    actions = envt.actions()
    
    #Run the environment for {ep_num} times.
    for ep in range(ep_num):
        
        #Initialize the start time & state.
        t0, n_sta = envt.mu()
        #Initialize the next time-state pair.
        nxt_t_sta = (t0, n_sta)
        #Iteration through the current episode.
        for t in range(t0, te):
            #The current state & time-state pair comes from the previous step or initialization.
            sta, t_sta = n_sta, nxt_t_sta
            #Choose one action.
            act = sel_act_(t_sta)
            #Take one step forward.
            (n_sta, rwd, ts) = envt.step(t, sta, act)
            #Generate the next time-state pair.
            nxt_t_sta = (t + 1, n_sta)
            #Calculate the new action-value.
            av, n_av = av_res.get_(t_sta, act), 0. if ts else (av_res.get_q_act_(nxt_t_sta))[1]
            av_new_val = av + alpha() * (rwd + envt.get_gamma(t, sta, act, n_sta) * n_av - av)
            #Update the current action-value.
            av_res.update_(t_sta, act, av_new_val)
            #Break the loop if the current episode is ended.
            if ts: break
            #inner-for-end
        #for-end
    #func-end


# ## Generalized Q-learning for NVMDPs

# In[ ]:


#Generalized Q-learning under NVMDPs.
def geq_learn(te, envt, sel_act_, ep_num, alpha, av_res):
    
    #Get all available actions.
    actions = envt.actions()
    #Get the number of action-value estimates minus one.
    n_minus_one = av_res.est_num - 1

    #Run the environment for {ep_num} times.
    for ep in range(ep_num):
        
        #Initialize the start time & state.
        t0, n_sta = envt.mu()
        #Initialize the next time-state pair.
        nxt_t_sta = (t0, n_sta)
        #Iteration through the current episode.
        for t in range(t0, te):
            #The current state & time-state pair comes from the previous step or initialization.
            sta, t_sta = n_sta, nxt_t_sta
            #Select the index of one action-value estimate track to update.
            est_ind = random.randint(0, n_minus_one)
            #Choose one action.
            act = sel_act_(t_sta, est_ind)
            #Take one step forward.
            (n_sta, rwd, ts) = envt.step(t, sta, act)
            #Generate the next time-state pair.
            nxt_t_sta = (t + 1, n_sta)
            #Get the current action-value estimate & the maximum representative action-value of the next state.
            av = av_res.get_est_(t_sta, act, est_ind)
            n_av = (av_res.get_q_act_(nxt_t_sta, est_ind))[1]
            #Calculate the new action-value estimate.
            av_new_val = av + alpha() * (rwd + envt.get_gamma(t, sta, act, n_sta) * n_av - av)
            #Store the new action-value estimate in the corresponding estimate track.
            av_res.update_est_(t_sta, act, est_ind, av_new_val)
            #Break the loop if the current episode is ended.
            if ts: break
            #inner-for-end
        #for-end


# In[ ]:





# In[ ]:




