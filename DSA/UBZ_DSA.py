#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Zhizuo Chen (aka George)
# 
# Permission is hereby granted, free of charge, to use, copy, modify, and distribute this code for any purpose, provided that this notice is included in all copies or substantial portions of the code.
# 
# This code is provided "as is", without warranty of any kind, express or implied. The author shall not be liable for any damages or consequences arising from its use.
# 
# This file contains implementations of algorithms and data structures inspired by 
# "Introduction to Algorithms" (CLRS), 3rd edition; however, all code is original and written in Python.
# 
# For questions or bug reports, contact: zhizuo.chen@outlook.com

# In[ ]:


import numpy as np
import pandas as pd
import time, math, random, scipy


# In[ ]:


#Global variables.
global GOLDEN_RATIO

#The golden ratio here is slightly smaller than its true value.
GOLDEN_RATIO = (1. + 5. ** .5) / 2. - 1e-6


# In[ ]:





# # Basic Tools

# In[ ]:


#Generate a geometric series which begins with {r^b} and ends with {r^n}.
def geom_ser(r, n, b = 0):
    
    num = n - b + 1
    prod = r ** b
    seq = [prod]
    for i in range(1, num):
        prod *= r
        seq.append(prod)
        #for-end
    return seq
    #func-end


# In[ ]:


#Generate a group of random numbers which sum to one.
def rand_sum2one(size):
    
    rand_prob = np.random.random(size)
    #Eliminate the theoretic probability that each random number equals zero.
    rand_prob[0] += 1e-10
    #Return the group of random numbers which sum to one.
    return np.divide(rand_prob, rand_prob.sum())
    #func-end


# In[ ]:


#Calculate the frequencies of 1-D array data.
def data_freq(data, intvl_size):
    
    #Set the lower & upper limit.
    a = math.floor(min(data) / intvl_size)
    b = math.floor(max(data) / intvl_size)
    intvl_ct = [0] * (b - a + 1)
    #Calculate the count of numbers in different intervals.
    for d in data: intvl_ct[math.floor(d / intvl_size) - a] += 1
    #Generate the interval lower bounds & corresponding frequencies.
    data_num = len(data)
    x = list()
    y = list()
    for i in range(len(intvl_ct)):
        ct = intvl_ct[i]
        if ct > 0:
            x.append((a + i) * intvl_size)
            y.append(ct / data_num)
            #if-end
        #for-end
    return [x, y]
    #func-end


# In[ ]:


#Sort the input data sequence in ascending order, and give the statistics of its duplicate values.
def sort_dupsta(seq):
    
    #The values in ascending order.
    seq_ = np.sort(seq).tolist()
    #The distinctive values & their counts.
    vals, cts = list(), list()
    #The current element & its distinctive value count.
    v, ct = seq_[0], 1

    #The main loop.
    for i in range(1, len(seq_)):
        #The next element.
        nv = seq_[i]
        #Check whether the current value is smaller than the next one.
        if v < nv:
            vals.append(v)
            cts.append(ct)
            ct = 1
        else: ct += 1
        #The next value becomes the current value in the next iteration.
        v = nv
        #for-end

    #Add the last element of the sorted sequence to the result.
    vals.append(v)
    cts.append(ct)
    #Return the sorted sequence, along with the statistics of its duplicate values.
    return [seq_, vals, cts]
    #func-end


# In[ ]:


#The class for scaling coordinates.
class Scaler:
    
    #The constructor.
    def __init__(self, old_bds, new_bds):

        #Get the dimension.
        dim = len(new_bds)
        #The dimensions of old coordinates and new coordinates must be equal.
        assert dim == len(old_bds), "!!Scaler.__init__.error: old and new dimensions are not equal!!"

        #The slopes & intercepts.
        slp, itc = list(), list()
        #Calculate the slopes & intercepts.
        for (a, b), (c, d) in zip(old_bds, new_bds):
            ba_diff = b - a
            slp.append((d - c) / ba_diff)
            itc.append((b * c - a * d) / ba_diff)
            #for-end
        #Store the slopes & intercepts.
        self.__slp = np.array(slp, dtype = np.float64)
        self.__itc = np.array(itc, dtype = np.float64)
        #func-end

    #Overload the "()" operation: transform the coordinates.
    def __call__(self, coords): return np.multiply(self.__slp, coords) + self.__itc
    #class-end


# In[ ]:





# # Partition Algorithms

# In[ ]:


#Rearranges the subarray in place such that the r-th element becomes the pivot element after transformation.
#See the pseudo-algorithm "partition" on "CLRS.3ed" P171.
def partn(arr, p, r, gt_f = lambda a, b: bool(a > b)):

    #Check whether the input of {p} & {r} make sense. 
    assert (p >= 0) and (p <= r) and (r <= len(arr) - 1), "Error: invalid values for p & r!!"
    #The starting index & the pivot.
    i, pvt = p - 1, arr[r]
    #The main loop.
    for j in range(p, r):
        #Swap elements if the current element is no greater than the pivot.
        if not gt_f(arr[j], pvt):
            i = i + 1
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            #if-end
        #for-end
    #Get the index of the leftmost element which is greater than the pivot.
    i1 = i + 1
    #Swap this leftmost element (greater than the pivot) with the pivot.
    if i1 < r:
        tmp = arr[i1]
        arr[i1] = arr[r]
        arr[r] = tmp
        #if-end
    #Return the index of the pivot element, which is within the range [p, r].
    return i1
    #func-end


# In[ ]:


#Rearranges the subarray in place with a given pivot.
#This algorithm is adapted from the pseudo-algorithm "partition" on "CLRS.3ed" P171.
def partn_adp(arr, p, r, pvt, gt_f = lambda a, b: bool(a > b)):

    #Check whether the input of {p} & {r} make sense. 
    assert (p >= 0) and (p <= r) and (r <= len(arr) - 1), "Error: invalid values for p & r!!"
    #Initialize the starting index.
    i = p - 1
    #The index of the maximum element in the left part after rearraning.
    lmax_ind, lmax_val = i, None
    #The main loop.
    for j in range(p, r + 1):
        #Swap elements if the current element is no greater than the pivot.
        if not gt_f(arr[j], pvt):
            i = i + 1
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            #Update the index of the maximum element in the left part.
            if (lmax_val is None) or gt_f(arr[i], lmax_val): lmax_ind, lmax_val = i, arr[i]
            #if-end
        #for-end
    #Make sure the rightmost element in the left part (if there is one) is the largest among the left elements.
    if (lmax_ind >= p) and (lmax_ind < i):
        tmp = arr[i]
        arr[i] = arr[lmax_ind]
        arr[lmax_ind] = tmp
        #if-end
    #Return the index of the rightmost element which is no greater than the pivot.
    #Notice that the range of {i} is [p-1, r] (or [p, r] if {pvt} is no less than the minimum of the subarray).
    return i
    #func-end


# ## Randomized-Select Algorithm

# In[ ]:


#Swap one element with the last element randomly, before applying "partition".
#See the pseudo-algorithm "random-partition" on "CLRS.3ed" P179.
def rand_partn(arr, p, r, gt_f = lambda a, b: bool(a > b)):

    #Randomly choose one element to swap with the last one.
    i = random.randint(p, r)
    #Swap this random chosen element with the last element.
    if i < r:
        tmp = arr[i]
        arr[i] = arr[r]
        arr[r] = tmp
        #if-end
    #Call "partition" & return the index of the pivot element.
    #Notice that the range of the return is [p, r].
    return partn(arr, p, r, gt_f)
    #func-end


# In[ ]:


#Return the i-th smallest element of the array using the Randomized-Partition procedure.
#See the pseudo-algorithm "randomized-select" on "CLRS.3ed" P216.
def rand_select(arr, p, r, i_th, gt_f = lambda a, b: bool(a > b)):

    #The answer is 100% certain if {p} & {r} are equal.
    if p == r: return arr[p]
    #Partition the sub-array with a random pivot.
    q = rand_partn(arr, p, r)
    #Check the relative order of {q}.
    k = q - p + 1
    #The pivot value is the answer.
    if k == i_th: return arr[q]
    #Search for the appropriate pivot value in the left part recursively.
    elif i_th < k: return rand_select(arr, p, q - 1, i_th, gt_f)
    #Search for the appropriate pivot value in the right part recursively.
    else: return rand_select(arr, q + 1, r, i_th - k, gt_f)
    #func-end


# ## Selection Algorithm

# In[ ]:


#Return the i-th smallest element of the array using the Selection algorithm (for real numbers only).
#The improvement of "selection" on "CLRS.3ed" P220.
def selection_rn(arr, p, r, i_th):
    
    #The answer is 100% certain if {p} & {r} are equal.
    if p == r: return arr[p]

    #The number of elements in the subarray.
    arr_len = r - p + 1
    #The remainder of {arr_len} divided by 5.
    arr_len5_rmdr = arr_len % 5
    #The maximum multiple of 5 which does not exceed {arr_len} & the sum of this number and the start index.
    arr_len5 = arr_len - arr_len5_rmdr
    r5 = p + arr_len5
    #The container of the medians.
    meds = list()
    #Search for the medians of every 5 elements.
    if arr_len5 > 0:
        arr5 = np.array(arr[p: r5], dtype = type(arr[0]))
        arr5 = arr5.reshape(-1, 5)
        meds.extend(np.sort(arr5, axis = 1)[:, 2])
    #Search for the median of the last few elements (4 elements at most).
    if arr_len5_rmdr > 0:
        arr5_rm_sort = np.sort(arr[r5: (r + 1)])
        meds.append(arr5_rm_sort[(arr_len5_rmdr - 1) // 2])
        #if-end
    
    #Use the median of the medians in {med} as the pivot.
    meds_num = len(meds)
    if meds_num <= 5: pvt = np.sort(meds)[(meds_num - 1) // 2]
    else: pvt = selection_rn(meds, 0, meds_num - 1, (meds_num + 1) // 2)
    
    #Partition the sub-array with the given pivot.
    q = partn_adp(arr, p, r, pvt)
    #Check the relative order of {q}.
    k = q - p + 1
    #The pivot value is the answer.
    if k == i_th: return arr[q]
    #Search for the appropriate pivot value in the left part recursively.
    elif i_th < k: return selection_rn(arr, p, q - 1, i_th)
    #Search for the appropriate pivot value in the right part recursively.
    else: return selection_rn(arr, q + 1, r, i_th - k)
    #func-end


# In[ ]:





# # Priority Queue

# In[ ]:


#This implement of priority queue uses the heap data structure.
class PriQ:
    
    def __init__(self, gt_f = None, init_q = None):
        
        #Implement a max-priority queue if {gt_f} is not specified.
        if gt_f is None: self.__gt_f = lambda x, y: bool(x > y)
        else: self.__gt_f = gt_f
            
        #Build a max heap from {init_q} if it isn't null.
        if init_q is not None:
            self.__q = list(init_q)
            for i in range((len(init_q) // 2) - 1, -1, -1): self.__heapify(i)
        #Otherwise an empty queue is created.
        else: self.__q = list()
        #func-end
    
    #Printable string representation.
    __str__ = lambda self: self.__q.__str__()
    #Length of the queue.
    __len__ = lambda self: self.__q.__len__()
    #The iterator.
    __iter__ = lambda self: self.__q.__iter__()
    #The index of the parent of the current node.
    ind_pa = lambda self, k: (k - 1) // 2
    #The index of the left child of the current node.
    ind_lc = lambda self, k: 2 * k + 1
    #The index of the left child of the current node.
    ind_rc = lambda self, k: 2 * k + 2
    #The element with maximum priority in the heap, which should be the first element.
    max_pri_ele = lambda self: None if 0 == len(self.__q) else self.__q[0]
    
    #!!Notice: this function assumes both the left and right subtrees of node {k} are properly organized heaps.
    #See the pseudo-algorithm "max-heapify" on "CLRS.3ed" P154.
    def __heapify(self, k):
        
        q_len = len(self.__q)
        #There is no need to heapify the queue if it is empty or has only one element.
        if q_len < 2: return
        lc = self.ind_lc(k)
        rc = self.ind_rc(k)
        lrgst = k
        
        if (lc < q_len) and self.__gt_f(self.__q[lc], self.__q[k]): lrgst = lc
        if (rc < q_len) and self.__gt_f(self.__q[rc], self.__q[lrgst]): lrgst = rc
        if lrgst != k:
            #Swap the largest element with the one indexed by {k}.
            tmp = self.__q[k]
            self.__q[k] = self.__q[lrgst]
            self.__q[lrgst] = tmp
            #Do the iteration here.
            self.__heapify(lrgst)
            #if-end
        #func-end
    
    #Pop the element with maximum priority in the queue.
    #See the pseudo-algorithm "heap-extract-max" on "CLRS.3ed" P163.
    def pop_max_pri(self):
        
        q_len = len(self.__q)
        if 0 == q_len: raise Exception("!!pop_max_pri.error: empty queue!!")
        if 1 == q_len: return self.__q.pop(0)
        
        max_ele = self.__q[0]
        self.__q[0] = self.__q.pop(-1)
        self.__heapify(0)
        return max_ele
        #func-end
    
    #Pop all the elements in order.
    def pop_all_ord(self):
        
        res = list()
        while len(self.__q) > 0: res.append(self.pop_max_pri())
        return res
        #func-end
    
    #If k is within the indices range of the queue, change node {k} to a new value with a greater priority.
    #Otherwise insert a new element with {new_value} into the queue.
    #See the pseudo-algorithms "heap-increase-key" & "max-heap-insert" on "CLRS.3ed" P164.
    def new_ele_val(self, k, new_val):
        
        q_len = len(self.__q)
        if (k >= 0) and (k < q_len): 
            if not self.__gt_f(new_val, self.__q[k]):
                raise Exception("!!incr_ele_val.error: a greater priority is needed!!")
            else: self.__q[k] = new_val
        else:
            self.__q.append(new_val)
            k = q_len
            #if-end

        pa = self.ind_pa(k)
        while (k > 0) and self.__gt_f(self.__q[k], self.__q[pa]):
            tmp = self.__q[pa]
            self.__q[pa] = self.__q[k]
            self.__q[k] = tmp
            k = pa
            pa = self.ind_pa(k)
            #while-end
        #func-end
    #class-end


# In[ ]:





# # The distribution of the max(X₁, X₂, ..., Xₙ)

# In[ ]:


#The distribution of the maximum of multiple independent discrete random variables.
def max_distr(val_grp, prob_grp):
    
    #The number of discrete distributions.
    m = len(val_grp)
    #The container of {value, prob, index}.
    vpi = list()
    for i in range(m):
        (va, pb) = (val_grp[i], prob_grp[i])
        for j in range(len(va)): vpi.append((va[j], pb[j], i))
        #for-end
        
    #Create a priority queue of {prob, value, index}.
    pq = PriQ(gtr_pri_f = lambda x, y: bool(x[0] < y[0]), init_q = vpi)
    #Get the elements of {prob, value, index} in the ascending order.
    vpi_asc = pq.pop_all_ord()
    #Add one "virtual maximum" element to {vpi}.
    vpi_asc.append((vpi_asc[-1][0] + 1., 0., 0))
    
    #The CDF & the number of variables that still have 0 CDF value.
    (cdf, z_cdf_ct) = ([0.] * m, m)
    #The number of all elements.
    ele_num = len(vpi_asc) - 1
    #The values & probabilities of the result.
    [res_val, res_prob] = [list(), list()]
    #Get the first element in the result with positive probability.
    (v, p, i) = vpi_asc[0]
    for pos in range(ele_num):
        (next_v, next_p, next_i) = vpi_asc[pos + 1]
        if (0. == cdf[i]) and (p > 0.): z_cdf_ct -= 1
        cdf[i] += p
        #Break the loop once the first element with positive pobability is found.
        if (0 == z_cdf_ct) and (v < next_v): break
        (v, p, i) = (next_v, next_p, next_i)
        #for-end
    #The probability of the current element in the result.
    curr_prob = np.prod(cdf)
    #Add the current value & probability to the result.
    res_val.append(v)
    res_prob.append(curr_prob)
    #The probability of the last observed element.
    prev_prob = curr_prob
    (v, p, i) = (next_v, next_p, next_i)
    for loc in range(pos + 1, ele_num):
        (next_v, next_p, next_i) = vpi_asc[loc + 1]
        #Update the CDF.
        prob_upd = cdf[i] + p
        curr_prob *= prob_upd / cdf[i]
        cdf[i] = prob_upd
        #Add the current value & probability to the result.
        if v < next_v:
            res_val.append(v)
            res_prob.append(curr_prob - prev_prob)
            prev_prob = curr_prob
            #if-end
        (v, p, i) = (next_v, next_p, next_i)
        #for-end
    #Return the probabilities & values.
    return [res_val, res_prob]
    #func-end


# In[ ]:





# # Red-Black Tree

# In[ ]:


#The class of Red-Black tree nodes.
class RBTree_Node:

    def __init__(self, val, *args):

        #The pointer to the parent, left child & right child.
        self.pa = None
        self.left = None
        self.right = None

        #Every new inserted node is dyed red.
        self.is_black = False
        #The value of the current node.
        self.val = val
        #func-end

    #Callable instance.
    __call__ = lambda self: self.val
    #Printable string representation.
    def __str__(self):
        strg = self.val.__str__() + ('#B' if self.is_black else '#R') + '\t'
        strg += self.pa.val.__str__() + '#' + self.left.val.__str__() + '#' + self.right.val.__str__()
        return strg
        #func-end
    #class-end


# In[ ]:


#The class of Red-Black Trees.
class RBTree:

    def __init__(self, gt_f = lambda a, b: bool(a > b), node_typ = RBTree_Node):
    
        #The type of nodes in the tree.
        self.node_typ = node_typ
        #The number of nodes of the current tree.
        self.__ct = 0
        #The leaf node is always black.
        self.__sntl = RBTree_Node(None)
        self.__sntl.is_black = True
        #The pointer to the root.
        self.__root = self.__sntl
        #The function that checks whether the first element is greater than the second element.
        self.__gt_f = gt_f
        #func-end

    #Get the number of nodes of the current tree.
    __len__ = lambda self: self.__ct
    #Get the pointer to the root of the tree.
    get_root = lambda self: self.__root if self.__ct > 0 else None
    #Get the minimum node value of the tree.
    min = lambda self: self.find_min().val if self.__ct > 0 else np.inf
    #Get the maximum node value of the tree.
    max = lambda self: self.find_max().val if self.__ct > 0 else -np.inf
    
    #Create a new node with a given value.
    def __new_node(self, val, *args):
        #Create a new node with the specified value.
        node = self.node_typ(val, *args)
        #Set its both children pointed to the leaf.
        node.left = node.right = self.__sntl
        #Return the pointer.
        return node
        #func-end
    
    #Rotate a non-leaf node to the left in the tree.
    #See the pseudo-algorithm "left-rotate" on "CLRS.3ed" P313.
    def __left_rota(self, x):
        #A pointer to {x}'s parent.
        x_pa = x.pa
        #Set {x}'s right child.
        y = x.right
        x.right = y.left
        if y.left is not self.__sntl: y.left.pa = x
        #Set {y}.
        y.pa = x_pa
        if x_pa is self.__sntl: self.__root = y
        elif x is x_pa.left: x_pa.left = y
        else: x_pa.right = y
        #Set {x}'s parent.
        y.left = x
        x.pa = y
        #func-end
    
    #Rotate a non-leaf node to the right in the tree.
    #See the pseudo-algorithm "left-rotate" on "CLRS.3ed" P313 (with "left" and "right" swapped).
    def __right_rota(self, x):
        #A pointer to {x}'s parent.
        x_pa = x.pa
        #Set {x}'s left child.
        y = x.left
        x.left = y.right
        if y.right is not self.__sntl: y.right.pa = x
        #Set {y}.
        y.pa = x_pa
        if x_pa is self.__sntl: self.__root = y
        elif x is x_pa.left: x_pa.left = y
        else: x_pa.right = y
        #Set {x}'s parent.
        y.right = x
        x.pa = y
        #func-end
    
    #Fix up the color issues after one new node is inserted.
    #See the pseudo-algorithm "rb-insert-fixup" on "CLRS.3ed" P316.
    def __rb_insert_fixup(self, z):
        #The main loop.
        while not z.pa.is_black:
            if z.pa is z.pa.pa.left:
                #The uncle.
                y = z.pa.pa.right
                #Case 1.
                if not y.is_black:
                    (z.pa.is_black, y.is_black, z.pa.pa.is_black) = (True, True, False)
                    z = z.pa.pa
                else:
                    #Case 2.
                    if z is z.pa.right:
                        z = z.pa
                        self.__left_rota(z)
                    #Case 3.
                    else:
                        (z.pa.is_black, z.pa.pa.is_black) = (True, False)
                        self.__right_rota(z.pa.pa)
                        #if-end
                    #if-end
            else:
                #The uncle.
                y = z.pa.pa.left
                #Case 1.
                if not y.is_black:
                    (z.pa.is_black, y.is_black, z.pa.pa.is_black) = (True, True, False)
                    z = z.pa.pa
                else:
                    #Case 2.
                    if z is z.pa.left:
                        z = z.pa
                        self.__right_rota(z)
                    #Case 3.
                    else:
                        (z.pa.is_black, z.pa.pa.is_black) = (True, False)
                        self.__left_rota(z.pa.pa)
                        #if-end
                    #if-end
            #while-end
        self.__root.is_black = True
        #func-end
    
    #Transplant {u} by {v}.
    #See the pseudo-algorithm "rb-transplant" on "CLRS.3ed" P323.
    def __rb_trsplt(self, u, v):
        #The parent of {u}.
        u_pa = u.pa
        #Check whether {u} is the root.
        if u_pa is self.__sntl: self.__root = v
        #Otherwise sets the corresponding child point of {u}'s parent to {v}.
        elif u is u_pa.left: u_pa.left = v
        else: u_pa.right = v
        #Reset {v}'s parent.
        v.pa = u_pa
        #func-end
    
    #Fix up the color issues after one node is deleted.
    #See the pseudo-algorithm "rb-delete-fixup" on "CLRS.3ed" P326.
    def __rb_delete_fixup(self, x):
        #The main loop.
        while (x is not self.__root) and x.is_black:
            #If x is its parent's left child.
            if x is x.pa.left:
                #The sibling of {x}.
                w = x.pa.right
                #Case 1.
                if not w.is_black:
                    (w.is_black, x.pa.is_black) = (True, False)
                    self.__left_rota(x.pa)
                    w = x.pa.right
                    #if-end
                #Case 2.
                if w.left.is_black and w.right.is_black: (w.is_black, x) = (False, x.pa)
                else:
                    #Case 3.
                    if w.right.is_black:
                        (w.left.is_black, w.is_black) = (True, False)
                        self.__right_rota(w)
                        w = x.pa.right
                        #if-end
                    #Case 4.
                    w.is_black = x.pa.is_black
                    (x.pa.is_black, w.right.is_black) = (True, True)
                    self.__left_rota(x.pa)
                    x = self.__root
                    #if-end
            #Otherwise x is its parent's right child.
            else:
                #The sibling of {x}.
                w = x.pa.left
                #Case 1.
                if not w.is_black:
                    (w.is_black, x.pa.is_black) = (True, False)
                    self.__right_rota(x.pa)
                    w = x.pa.left
                    #if-end
                #Case 2.
                if w.left.is_black and w.right.is_black: (w.is_black, x) = (False, x.pa)
                else:
                    #Case 3.
                    if w.left.is_black:
                        (w.right.is_black, w.is_black) = (True, False)
                        self.__left_rota(w)
                        w = x.pa.left
                        #if-end
                    #Case 4.
                    w.is_black = x.pa.is_black
                    (x.pa.is_black, w.left.is_black) = (True, True)
                    self.__right_rota(x.pa)
                    x = self.__root
                    #if-end
                #if-end
            #while-end
        #Set {x}'s color to black.
        x.is_black = True
        #func-end
    
    #Inorder tree walk.
    #See the pseudo-algorithm "inorder-tree-walk" on "CLRS.3ed" P288.
    def inorder_walk(self, x = None):
        #Inorderly walk through the whole tree if {x} is NULL.
        if x is None: x = self.__root
        #The main loop.
        if x is not self.__sntl:
            self.inorder_walk(x.left)
            print(x)
            self.inorder_walk(x.right)
            #if-end
        #func-end.
    
    #Fetch the pointers to nodes in the whole tree by their values' ascending order, without using stacks.
    def fetch_all_asc(self):
        
        #The pointer to the root, the id of the current node and the id of the next node.
        (ptr, x_id, n_id) = (self.__root, 0, 1)
        #The parents' id of all nodes.
        pa_grp = [-1] * self.__ct
        #The statuses of all nodes, namely the counts of upward visits to the nodes.
        sts_grp = [0] * self.__ct
        #The remaining number of unfetched values and the container of the fetched values.
        rm_ct = self.__ct
        #The containers of all node values & pointers by their values' ascending order.
        (val_grp, ptr_grp) = (list(), list())
        #Fetch all values from all nodes.
        while rm_ct > 0:
            #Check the status of the current node.
            sts = sts_grp[x_id]
            #If it is the first time to visit the current node.
            if 0 == sts:
                if ptr.left is self.__sntl: sts_grp[x_id] = 1
                else:
                    pa_grp[n_id] = x_id
                    (ptr, x_id) = (ptr.left, n_id)
                    n_id += 1
                    #if-end
            #Or only the left subtree of the current node has been traversed.
            elif 1 == sts:
                val_grp.append(ptr.val)
                ptr_grp.append(ptr)
                rm_ct -= 1
                if ptr.right is self.__sntl: sts_grp[x_id] = 2
                else:
                    pa_grp[n_id] = x_id
                    (ptr, x_id) = (ptr.right, n_id)
                    n_id += 1
                    #if-end
            #Otherwise return to the parent if both subtrees of the current node have been traversed.
            else:
                (ptr, x_id) = (ptr.pa, pa_grp[x_id])
                sts_grp[x_id] += 1
            #while-end
        return (val_grp, ptr_grp)
        #func-end
    
    #Return the pointer to the node with minimum value.
    #See the pseudo-algorithm "tree-minimum" on "CLRS.3ed" P291.
    def find_min(self, node = None):
        #Return null if the tree has zero nodes.
        if 0 == self.__ct: return None
        #Initialize the pointers.
        ptr_pa = self.__root if node is None else node
        ptr = ptr_pa.left
        #Find the leftmost node.
        while ptr is not self.__sntl:
            ptr_pa = ptr
            ptr = ptr.left
            #while-end
        return ptr_pa
        #func-end
    
    #Return the pointer to the node with maximum value.
    #See the pseudo-algorithm "tree-maximum" on "CLRS.3ed" P291.
    def find_max(self, node = None):
        #Return null if the tree has zero nodes.
        if 0 == self.__ct: return None
        #Initialize the pointers.
        ptr_pa = self.__root if node is None else node
        ptr = ptr_pa.right
        #Find the rightmost node.
        while ptr is not self.__sntl:
            ptr_pa = ptr
            ptr = ptr.right
            #while-end
        return ptr_pa
        #func-end
    
    #Return the pointer to one node with a specified value.
    #See the pseudo-algorithm "tree-search" on "CLRS.3ed" P290.
    def find(self, val):
        #Initialize the pointer.
        ptr = self.__root
        #Try to find a node with the specified value until reaching a leaf.
        while ptr is not self.__sntl:
            node_val = ptr()
            if self.__gt_f(val, node_val): ptr = ptr.right
            elif self.__gt_f(node_val, val): ptr = ptr.left
            else: break
            #while-end
        return (None if ptr is self.__sntl else ptr)
        #func-end

    #Insert one new node with a specified value into the tree.
    #See the pseudo-algorithm "rb-insert" on "CLRS.3ed" P315.
    def rb_insert(self, val, *args):
        #The pointer to the new node.
        node = self.__new_node(val, *args)
        #The temporary variables.
        (x, y, go_left) = (self.__root, self.__sntl, True)

        #Traverse the tree until reaching a leaf.
        while x is not self.__sntl:
            y = x
            if self.__gt_f(x(), val):
                x = x.left
                go_left = True
            else:
                x = x.right
                go_left = False
                #if-end
            #while-end
        #Update the new inserted node.
        node.pa = y
        if y is self.__sntl: self.__root = node
        elif go_left: y.left = node
        else: y.right = node
        #Maintain the properties of the RB tree.
        self.__rb_insert_fixup(node)
        #The number of nodes increases by one.
        self.__ct += 1
        
        #Return the pointer to the newly created node.
        return node
        #func-end
    
    #Delete one node being pointed from the tree.
    #See the pseudo-algorithm "rb-delete" on "CLRS.3ed" P324.
    def rb_delete_ptr(self, z):
        #Avoid deleting "null" nodes.
        if (z is None) or (z is self.__sntl): return
        #Replicate the pointer to {z} and copy its original color.
        (y, ori_color) = (z, z.is_black)
        #If {y} has no more than one sub-branch.
        if z.left is self.__sntl:
            x = z.right
            self.__rb_trsplt(z, x)
        elif z.right is self.__sntl:
            x = z.left
            self.__rb_trsplt(z, x)
        #Otherwise find the node with minimum value in the right subtree.
        else:
            (z_left, z_right) = (z.left, z.right)
            y = self.find_min(z_right)
            (ori_color, x) = (y.is_black, y.right)
            #If {z.right} has no left branch.
            if y.pa is z: x.pa = y
            #Otherwise replace {y} with its right branch.
            else:
                self.__rb_trsplt(y, x)
                (y.right, z_right.pa) = (z_right, y)
                #if-end
            #Replace {z} with {y}.
            self.__rb_trsplt(z, y)
            (y.left, z_left.pa, y.is_black) = (z_left, y, z.is_black)
            #if-end
        #If the original color of {y} is black then fix up the tree.
        if ori_color is True: self.__rb_delete_fixup(x)
        #The number of nodes decreases by one.
        self.__ct -= 1
        #func-end
        
    #Delete one node with a specified value from the tree.
    def rb_delete(self, val):
        #Find the node with the specified value.
        node = self.find(val)
        #Delete the node with the specified value once it has been found.
        if node is not None: self.rb_delete_ptr(node)
        return node
        #func-end
    #class-end


# In[ ]:





# # Doubly Linked Lists

# In[3]:


#The class of doubly linked list node.
class DLL_Node:
    
    def __init__(self, val):
        
        #The pointer to the left & right node.
        (self.left, self.right) = (None, None) 
        #The value of the current node.
        self.val = val
    #class-end
    
#The class of doubly linked list node, for usage in Fibonacci Heaps.
class FH_DLL_Node(DLL_Node):
    
    def __init__(self, val):
        
        #Call the parent classâ€™s initializing method to initialize the inherited attributes.
        super().__init__(val)
        #The pointer to the parent & the pointer to child doubly linked list.
        (self.pa, self.chd) = (None, DLL(fh_flag = True))
        #Whether this node has lost a child since the last time it was made the child of another node.
        self.mark = False
        #func-end
    
    #Get the degree (i.e. the number of children).
    degree = lambda self: len(self.chd)
    #class-end


# In[4]:


#The class of circular doubly linked lists.
class DLL:
    
    def __init__(self, fh_flag):
        
        #The type of nodes in the doubly linked list.
        self.node_typ = FH_DLL_Node if fh_flag else DLL_Node
        #The pointer to the head of the doubly linked list.
        self.head = None
        #The number of nodes in the doubly linked list.
        self.__ct = 0
        #func-end
    
    #Clear all the nodes & reset the node count.
    def clear(self): (self.head, self.__ct) = (None, 0)
    
    #Store all the nodes in a list.
    def tolist(self):
        
        #The container of pointers to all nodes in the doubly linked list.
        nodes = list()
        #Get the head of the doubly linked list.
        x = self.head
        #Add each node's value to the list, starting from the head.
        for i in range(self.__ct):
            nodes.append(x)
            x = x.right
            #for-end
        #Return the result.
        return nodes
        #func-end
    
    #Get the length of the doubly linked list.
    __len__ = lambda self: self.__ct
    #Print the nodes value in the doubly linked list.
    def __str__(self):
        res_str = ""
        nodes = self.tolist()
        for node in nodes: res_str += node.val.__str__() + ', '
        return res_str[: -2]
        #func-end
    
    #Insert a new node with a given value in the right side of a specified node.
    def insert(self, v, z = None):
        
        #Create a new node if {v} is not a pointer to a node.
        node = v if isinstance(v, DLL_Node) else self.node_typ(v)
        #The node becomes the head if the current doubly linked list is empty.
        if self.head is None:
            self.head = node
            (node.left, node.right) = (node, node)
        #Otherwise insert the node in the right side of the specified node (or tail if no specified).
        else:
            if z is None: z = self.head.left
            zr = z.right
            #Update the left & right pointers of the new node.
            (node.left, node.right) = (z, zr)
            #Update the right pointer of {z} & the left pointer of {z}'s original right node.
            (z.right, zr.left) = (node, node)
            #if-end
        
        #The number of nodes increases by one.
        self.__ct += 1
        #Return the pointer to the newly created node.
        return node
        #func-end
    
    #Merge with another doulby linked list.
    def merge(self, dll2):
        
        #The two lists should be the same type.
        assert self.node_typ is dll2.node_typ, "Error: merging lists of different types!!"
        #Nothing to do if the other doubly linked list is the current one or empty.
        if (dll2 is self) or (dll2.head is None): return
        #If the current doubly linked list is empty, then modify its pointers & nodes count.
        elif self.head is None:
            self.head = dll2.head
            #Update the number of nodes.
            self.__ct = len(dll2)
            #Empty {dll2} after the merging.
            dll2.clear()
        #Append {dll2} to the end of the current doubly linked list if both lists aren't empty.
        else:
            #The tails of the two doubly linked lists.
            (tail, tail2) = (self.head.left, dll2.head.left)
            #Link the two lists.
            (tail.right, dll2.head.left) = (dll2.head, tail)
            #Modify the tail of the merge doubly linked list.
            (tail2.right, self.head.left) = (self.head, tail2)
            #Update the number of nodes.
            self.__ct += len(dll2)
            #Empty {dll2} after merging.
            dll2.clear()
            #if-end
        #func-end
    
    #Delete one node with a given pointer from the doubly linked list.
    def delete_ptr(self, z):
        
        #The doubly linked list should not be empty before deletion.
        assert self.__ct > 0, "Error: the doubly linked list is empty!!"
        
        #Modify the left & right nodes of {z}.
        (zl, zr) = (z.left, z.right)
        (zl.right, zr.left) = (zr, zl)
        #Reset the head to null if there is no nodes left after deletion.
        if 1 == self.__ct: self.head = None
        #Otherwise reset the head if {z} is the current head.
        elif z is self.head: self.head = zr
        
        #The number of nodes decreases by one.
        self.__ct -= 1
        #func-end
    #class-end


# In[ ]:





# # Fibonacci Heaps

# In[5]:


#The class of Fibonacci heaps.
class FibH:
    
    def __init__(self, gt_f = None, init_q = None):
        
        #The total number of nodes in the Fibonacci heap.
        self.__nodes_ct = 0
        
        #The Fibonacci heap is max-heap ordered if {__gt_f} is not specified.
        if gt_f is None: self.__gt_f = lambda x, y: bool(x > y)
        else: self.__gt_f = gt_f
        #The node with maximum priority.
        self.m_node = None
        
        #The roots form a circular doubly linked list.
        self.roots = DLL(fh_flag = True)
        #Initialize roots from {init_q} if it isn't null.
        if init_q is not None:
            #Insert the first element in {init_q}.
            m_val = init_q[0]
            self.m_node = self.roots.insert(m_val)
            #Insert the remaining elements in {init_q}.
            for val in init_q[1: ]:
                node = self.roots.insert(val)
                if self.__gt_f(val, m_val): (m_val, self.m_node) = (val, node)
                #for-end
            self.__nodes_ct = len(init_q)
            #if-end
        #func-end
        
    #Length of the Fibonacci heap.
    __len__ = lambda self: self.__nodes_ct
    #The node with maximum priority in the Fibonacci heap.
    max_pri_ele = lambda self: None if 0 == self.__nodes_ct else self.m_node.val
    
    #Print the nodes in the Fibonacci heap.
    def __str__(self):
        #The result string for display.
        res_str = "Nodes: %d\nRoots:  " % self.__nodes_ct
        #The temporary container of nodes with children.
        parents = list()
        #Loop through all nodes in the roots.
        node = self.roots.head
        for i in range(len(self.roots)):
            res_str += node.val.__str__() + ', '
            if node.degree() > 0: parents.append((node, 1))
            node = node.right
            #for-end
        res_str = res_str[: -2] + '\n'
        #Loop through all nodes with children.
        while len(parents) > 0:
            (node, lev) = parents.pop()
            #The level of the children should increase by one.
            lev_plus_1 = lev + 1
            res_str += "Pa: " + node.val.__str__() + (" Lev %d:  " % lev)
            #Loop through the chidlren of the current node.
            x = node.chd.head
            for i in range(node.degree()):
                res_str += x.val.__str__() + ', '
                if x.degree() > 0: parents.append((x, lev_plus_1))
                x = x.right
                #for-end
            res_str = res_str[: -2] + '\n'
            #while-end
        return res_str
        #func-end
    
    #Reduce the number of trees in the Fibonacci heap by the consolidating process.
    #See the pseudo-algorithm "consolidate" on "CLRS.3ed" P516.
    def __consolidate(self):
        
        #The upper bound of degree. 
        degr_upp = math.floor(math.log(self.__nodes_ct, GOLDEN_RATIO))
        #A new array with {degr_upp} + 1 elements, all intialized to null.
        A = [None] * (degr_upp + 1)
        
        #The flag whether the iteration should stop after the current one.
        loop_cont_flag = True
        #The iteration starts from the head of the roots and ends after dealing with the "tail".
        x = self.roots.head
        last_node = x.left
        #Loop through all nodes in the roots.
        while loop_cont_flag:
            #Check whether the iteration should end after the current one.
            if x is last_node: loop_cont_flag = False
            #Get the degree of the current node & store the pointer to the next node.
            (d, nx) = (x.degree(), x.right)
            #Check whether there is another recorded node with the same degree.
            while A[d] is not None:
                #The other node with the same degree, which has been recorded.
                y = A[d]
                #Exchange the pointers of {x} & {y} if {y} has a higher priority.
                if self.__gt_f(y.val, x.val):
                    tmp = x
                    x = y
                    y = tmp
                    #if-end
                #---- ----- the procedure of "fib-heap-link" on "CLRS.3ed" P516 ---- ---- 
                #Remove {y} from the roots.
                self.roots.delete_ptr(y)
                #Make {y} a child of {x}.
                x.chd.insert(y)
                y.pa = x
                #Update the mark of {y}.
                y.mark = False
                #---- ----- ---- ---- ---- ----- ---- ---- ---- ----- ---- ---- 
                A[d] = None
                d += 1
                #inner-while-end
            #Reset the pointer stored in {A[d]}.
            A[d] = x
            #Iterate to the next node.
            x = nx
            #outer-while-end
        
        """
        #---- ----- The procedure here corresponds to line 15-23 of the original algorithm on "CLRS.3ed" P516, 
        #which is replaced by the following lines. ---- ---- 
        #Intialize the node with maximum priority to null.
        self.m_node = None
        #Loop through all node pointers stored in {A}.
        for i in range(len(A)):
            if A[i] is not None:
                #If encounter a non-null element in {A} for the first time.
                if self.m_node is None:
                    #Create a root list for H containing just {A[i]}.
                    self.roots = DLL(fh_flag = True)
                    self.roots.insert(A[i])
                    #Reset the node with maximum priority.
                    self.m_node = A[i]
                #Otherwise insert {A[i]} into the new roots.
                else:
                    self.roots.insert(A[i])
                    #Update the node with maximum priority.
                    if self.__gt_f(A[i].val, self.m_node.val): self.m_node = A[i]
                #if-end
            #for-end
        #---- ----- ---- ---- ---- ----- ---- ---- ---- ----- ---- ---- 
        """
        #Reset the node with maximum priority to the roots' current head.
        self.m_node = self.roots.head
        #Loop through all the remaining nodes in the roots and update the node with maximum priority.
        x = self.m_node.right
        for i in range(len(self.roots) - 1):
            #Update the node with maximum priority.
            if self.__gt_f(x.val, self.m_node.val): self.m_node = x
            #Iterate to the next node.
            x = x.right
            #for-end    
        #func-end
    
    #Clear all the nodes & reset the total node count.
    def clear(self):
        self.roots.clear()
        self.__nodes_ct = 0
        #func-end
    
    #Insert a new node with a given value into the Fibonacci heap.
    #See the pseudo-algorithm "fib-heap-insert" on "CLRS.3ed" P510.
    def insert(self, val):
        
        #The new node with the given value will be placed in roots.
        node = self.roots.insert(val)
        #Check whether this new node has a higher priority.
        if (self.m_node is None) or self.__gt_f(val, self.m_node.val): self.m_node = node        
        #The total number of nodes increases by one.
        self.__nodes_ct += 1
        #Return the pointer to the newly inserted node.
        return node
        #func-end
    
    #Merge with another Fibonacci heap.
    #See the pseudo-algorithm "fib-heap-union" on "CLRS.3ed" P512.
    def merge(self, fh2):
        
        #The current total nodes count of {fh2}.
        fh2_len = len(fh2)
        #Merge the current root list with the other one.
        self.roots.merge(fh2.roots)
        #Update the node with maximum priority.
        if (self.m_node is None) or ((fh2_len > 0) and self.__gt_f(fh2.m_node.val, self.m_node.val)): 
            self.m_node = fh2.m_node
            #if-end
        #Update the total number of nodes.
        self.__nodes_ct += fh2_len
        #Empty {fh2} after merging.
        fh2.clear()
        #func-end
    
    #Pop the node with maximum value in the Fibonacci heap.
    #See the pseudo-algorithm "fib-heap-extract-min" on "CLRS.3ed" P513.
    def pop_max_pri(self):
        
        #{z} points to the node with maximum priority.
        z = self.m_node
        if z is not None:
            #Extract each child of {z} and add it to the roots.
            z_chd = z.chd.head
            for i in range(z.degree()):
                z_chd_r = z_chd.right
                #Add the child to the roots & modify the child's parent pointer.
                self.roots.insert(z_chd)
                z_chd.pa = None
                #Iterate to the next child.
                z_chd = z_chd_r
                #for-end
            z.chd.clear()
            #Delete {z} from the roots.
            self.roots.delete_ptr(z)
            #Reset the node with maximum priority to null if the roots become empty.
            if 0 == len(self.roots): self.m_node = None
            #Otherwise reset the node with maximum priority.
            else:
                self.m_node = z.right
                self.__consolidate()
                #if-end
            #The total number of nodes decreases by one.
            self.__nodes_ct -= 1
            #if-end
        return z
        #func-end
    
    #Pop all the nodes with their values in order.
    def pop_all_ord(self):
        
        res = list()
        while self.__nodes_ct > 0: res.append(self.pop_max_pri().val)
        return res
        #func-end
    
    #Alter the value of a node (pointed by {x}) to {new_val}, which has higher priority.
    #See the pseudo-algorithm "fib-heap-decrease-key" on "CLRS.3ed" P519.
    def alter_key_val(self, x, new_val):
        
        #{new_val} shouldn't be lower in priority compared with the current one.
        assert self.__gt_f(new_val, x.val) or (not self.__gt_f(x.val, new_val)), \
            "Error: the new value should have a higher priority!!"
        #Replace the current value by {new_val}.
        x.val = new_val
        #Get {x}'s parent.
        y = x.pa
        
        #Restore the heap order if it is violated after {x}'s value has been udpated.
        if (y is not None) and (self.__gt_f(new_val, y.val) or (not self.__gt_f(y.val, new_val))):
            #Copy the pointer of {x} to {z} before iteration.
            z = x
            #Loop {z} upwardly till reaching a node in the roots or a false-marked node.
            while True:
                #---- ----- the procedure of "cut" on "CLRS.3ed" P519 ---- ---- 
                #Remove {z} from {y}'s children.
                y.chd.delete_ptr(z)
                #Add {z} to the roots.
                self.roots.insert(z)
                #Update the parent & the mark of {z}.
                (z.pa, z.mark) = (None, False)
                #---- ----- ---- ---- ---- ----- ---- ---- ---- ----- ---- ---- 
                #---- ----- the procedure of "cascading-cut" on "CLRS.3ed" P519 ---- ---- 
                #The restoration process ends if {y} has no parent (i.e. {y} is in the roots).
                if y.pa is None: break
                #If {y}'s mark attribute is false, reverse it & quit the restoration process.
                elif not y.mark:
                    y.mark = True
                    break
                #Otherwise change {y} to its parent for the next iteration.
                else:
                    z = y
                    y = y.pa
                    #if-end
                #---- ----- ---- ---- ---- ----- ---- ---- ---- ----- ---- ---- 
                #while-end
            #if-end
        #Update the node with maximum priority.
        if (self.__gt_f(new_val, self.m_node.val) or (not self.__gt_f(self.m_node.val, new_val))): self.m_node = x
        #func-end
    
    #Deleting a node (pointed by {x}).
    #See the pseudo-algorithm "fib-heap-delete" on "CLRS.3ed" P522.
    def delete_ptr(self, x):
        #Change node {x}'s value to the value with maximum priority in the Fibonacci heap.
        self.alter_key_val(x, self.m_node.val)
        #Kick node {x} off.
        self.pop_max_pri()
        #func-end
    #class-end


# In[ ]:





# # reduced-space van Emde Boas Trees

# In[ ]:


#The class of reduced-space van Emde Boas tree structure.
class rs_vEB_struct:

    #The input {lgu} must be an integer.
    def __init__(self, lgu):

        #Check whether {lgu} is smaller than 1.
        assert lgu >= 1, "!!rs_vEB_struct __init__ error: lgu is smaller than 1!!"
        
        #The logarithm to the base 2 of the universe size u.
        self.lgu = lgu
        #The size of the least significant bits.
        self.__low_bits = lgu // 2
        #The size of the most significant bits & the value of ↓√u.
        (self.__upp_bits, self.d_sqrt_u) = (lgu - self.__low_bits, 2 ** self.__low_bits)
        #The pointers to the minimum & maximum elements.
        (self.veb_min, self.veb_max) = (None, None)
        
        #The structure contains a summary & a cluster if the universe size is larger than 2.
        if lgu > 1:
            #The pointer to the summary.
            self.summary = None
            #The cluster is a hash table which stores ↑√u vEB(↓√u) sub-trees at most.
            self.cluster = dict()
            #if-end
        #func-end
    
    #Functions for operating the input integer(s). See "high", "low" & "index" on "CLRS.3ed" P546.
    high = lambda self, x: x // self.d_sqrt_u
    low = lambda self, x: x % self.d_sqrt_u
    index = lambda self, x, y: x * self.d_sqrt_u + y
    #Get the minimum element. See "vEB-tree-minimum" on "CLRS.3ed" P550.
    min = lambda self: self.veb_min
    #Get the maximum element. See "vEB-tree-maximum" on "CLRS.3ed" P550.
    max = lambda self: self.veb_max
    
    #Check whether integer {x} (which must be within [0, 2 ** {self.lgu} - 1]) is in the rs-vEB tree.
    #Adapted from the pseudo-algorithm "vEB-tree-member" on "CLRS.3ed" P550.
    def member(self, x):
        if (x == self.veb_min) or (x == self.veb_max): return True
        elif 1 == self.lgu: return False
        else:
            chd = self.cluster.get(self.high(x))
            if chd is None: return False
            else: return chd.member(self.low(x))
            #if-end
        #func-end
    
    #List all stored integers.
    def all_ints(self):
        
        #The container of the result.
        ints = list()
        #Return an empty list if the rs-vEB tree is empty.
        if self.veb_min is None: return ints
        
        #Add the minimum element to the (result) container.
        ints.append(self.veb_min)
        #Check whether the rs-vEB tree has one sole element.
        if self.veb_max == self.veb_min: pass
        #Return both the minimum & the maximum elements if they aren't equal for the case of rs-vEB(2).
        elif 1 == self.lgu: ints.append(self.veb_max)
        #Otherwise retrieve all elements in a recursive manner.
        else:
            #Loop through all sub-trees in {self.cluster}.
            for k, chd in self.cluster.items():
                chd_ints = chd.all_ints()
                if len(chd_ints) > 0: ints.extend(np.add(k * self.d_sqrt_u, chd_ints))
                #for-end
            #if-end
        #Return the result (all currently stored integers).
        return ints
        #func-end
    
    #Insert integer {x} (which must be within [0, 2 ** {self.lgu} - 1] and not in the tree currently) into the rs-vEB tree.
    #See the pseudo-algorithm "vEB-tree-insert" on "CLRS.3ed" P553.
    def insert(self, x):
        #Check whether the tree structure has no elements before insertion.
        if self.veb_min is None: (self.veb_min, self.veb_max) = (x, x)
        else: 
            #Swap {x} with the current minimum if {x} is smaller.
            if x < self.veb_min:
                tmp = x
                x = self.veb_min
                self.veb_min = tmp
                #if-end
            #Insert {x} into the corresponding cluster recursively & update the summary if necessary.
            if self.lgu > 1:
                #The temporary values of ↑√x & ↓√x.
                (high_x, low_x) = (self.high(x), self.low(x))
                #{chd} points to the cluster that needs to be updated.
                chd = self.cluster.get(high_x)
                #Create the rs-vEB sub-tree if it hasn't been created.
                if chd is None:
                    chd = self.__class__(self.__low_bits)
                    self.cluster[high_x] = chd
                    #if-end
                #Update {chd} if no element has been stored in it.
                if chd.min() is None:
                    #Assign a rs-vEB sub-tree with an universe size of ↑√u to {self.summary} if it is null.
                    if self.summary is None: self.summary = self.__class__(self.__upp_bits)
                    self.summary.insert(high_x)
                    #Update the minimum & maximum elements of {chd}.
                    (chd.veb_min, chd.veb_max) = (low_x, low_x)
                #Otherwise store a new element in {chd}.
                else: chd.insert(low_x)
                #if-end
            #Update the current maximum if {x} is larger than it.
            if x > self.veb_max: self.veb_max = x
            #if-end
        #func-end
        
    #Delete integer {x} (which must be within [0, 2 ** {self.lgu} - 1] and in the tree currently) from the rs-vEB tree.
    #See the pseudo-algorithm "vEB-tree-delete" on "CLRS.3ed" P554.
    def delete(self, x):
        #Reset the minimum & the maximum to null if the rs-vEB tree stores one sole integer.
        if self.veb_min == self.veb_max: (self.veb_min, self.veb_max) = (None, None)
        #Delete one of the two integers for the case of a rs-vEB(2) tree.
        elif 1 == self.lgu:
            if 0 == x: self.veb_min = 1
            else: self.veb_min = 0
            self.veb_max = self.veb_min
        #Delete one element from multiple stored elements in a rs-vEB tree larger than rs-vEB(2).
        else:
            #Update the minimum & replace {x} with the second-smallest integer if {x} equals the minimum.
            if self.veb_min == x:
                first_cluster = self.summary.veb_min
                x = self.index(first_cluster, self.cluster[first_cluster].veb_min)
                self.veb_min = x
                #if-end
            #The temporary values of ↑√x & ↓√x, and the pointer to {x}'s sub-tree.
            (high_x, low_x) = (self.high(x), self.low(x))
            chd = self.cluster[high_x]
            #Delete {x} from the cluster recursively.
            chd.delete(low_x)
            #Update the summary if {x}'s cluster becomes empty after the deletion.
            if chd.veb_min is None:
                self.summary.delete(high_x)
                #Update the rs-vEB's maximum element if {x} equals it.
                if self.veb_max == x:
                    summary_max = self.summary.veb_max
                    if summary_max is None: self.veb_max = self.veb_min
                    else: self.veb_max = self.index(summary_max, self.cluster[summary_max].veb_max)
                    #if-end
                #if-end
            #Update the rs-vEB's maximum element if {x} equals it and {x}'s sub-tree isn't empty after deletion.
            elif self.veb_max == x: self.veb_max = self.index(high_x, chd.veb_max)
            #if-end
        #func-end

    #Find the predecessor of integer {x} (which must be within [0, 2 ** {self.lgu} - 1]) from the rs-vEB tree.
    #See the pseudo-algorithm "vEB-tree-predecessor" on "CLRS.3ed" P552.
    def predecessor(self, x):
        #Find the predecessor of {x} for the case of a rs-vEB(2) tree.
        if 1 == self.lgu:
            #{x} will have one predecessor only if it is 1 and the minimum is 0.
            if (1 == x) and (0 == self.veb_min): return 0
            else: return None
        #The predecessor is the current maximum if {x} is larger than it.
        elif (self.veb_max is not None) and (x > self.veb_max): return self.veb_max
        #Otherwise find the predecessor iteratively.
        else:
            #The temporary values of ↑√x & ↓√x, and the pointer to {x}'s sub-tree.
            (high_x, low_x) = (self.high(x), self.low(x))
            chd = self.cluster.get(high_x)
            #Find the minimum element in {x}'s sub-tree.
            min_low = None if chd is None else chd.veb_min
            #Check whether {x}'s sub-tree has a valid predecessor.
            if (min_low is not None) and (low_x > min_low): return self.index(high_x, chd.predecessor(low_x))
            #Otherwise find the predecessor sub-tree of {x}'s sub-tree.
            else:
                #Check whether {self.summary} is initialized.
                if self.summary is not None: pred_cluster = self.summary.predecessor(high_x)
                else: pred_cluster = None
                #Check whether the predecessor of {x} is the current minimum if {pred_cluster} is null.
                if pred_cluster is None:
                    if (self.veb_min is not None) and (x > self.veb_min): return self.veb_min
                    else: return None
                #Otherwise find the predecessor through {x}'s predecessor sub-tree.
                else: return self.index(pred_cluster, self.cluster[pred_cluster].veb_max)
                #if-end
            #if-end
        #func-end
        
    #Find the successor of integer {x} (which must be within [0, 2 ** {self.lgu} - 1]) from the rs-vEB tree.
    #See the pseudo-algorithm "vEB-tree-successor" on "CLRS.3ed" P551.
    def successor(self, x):
        #Find the successor of {x} for the case of a rs-vEB(2) tree.
        if 1 == self.lgu:
            #{x} will have one successor only if it is 0 and the maximum is 1.
            if (0 == x) and (1 == self.veb_max): return 1
            else: return None
        #The successor is the current minimum if {x} is smaller than it.
        elif (self.veb_min is not None) and (x < self.veb_min): return self.veb_min
        #Otherwise find the successor iteratively.
        else:
            #The temporary values of ↑√x & ↓√x, and the pointer to {x}'s sub-tree.
            (high_x, low_x) = (self.high(x), self.low(x))
            chd = self.cluster.get(high_x)
            #Find the maximum element in {x}'s sub-tree.
            max_low = None if chd is None else chd.veb_max
            #Check whether {x}'s sub-tree has a valid successor.
            if (max_low is not None) and (low_x < max_low): return self.index(high_x, chd.successor(low_x))
            #Otherwise find the successor sub-tree of {x}'s sub-tree.
            else:
                #Check whether {self.summary} is initialized.
                if self.summary is None: return None
                else: succ_cluster = self.summary.successor(high_x)
                #Check whether {x} has a successor sub-tree.
                if succ_cluster is None: return None
                else: return self.index(succ_cluster, self.cluster[succ_cluster].veb_min)
                #if-end
            #if-end
        #func-end
    #class-end


# In[ ]:


#The class of reduced-space van Emde Boas tree.
class rs_vEB_tree:
    
    def __init__(self, lgu = 1):
        
        #The root is a rs-vEB tree structure with a specified universe size.
        self.__root = rs_vEB_struct(lgu)
        #The range of integers of the rs-vEB tree is [0, 2^{lgu} - 1].
        self.__max_int = 2 ** lgu - 1
        #The count of elements stored.
        self.__ct = 0
        #func-end
    
    #Reload the "length" method.
    __len__ = lambda self: self.__ct
    #Reload the "print" method.
    __str__ = lambda self: self.__root.all_ints().__str__()
    #Show the integer of the upper bound that could be stored in this rs-vEB tree.
    max_int = lambda self: self.__max_int
    #Get the minimum element.
    min = lambda self: self.__root.min()
    #Get the maximum element.
    max = lambda self: self.__root.max()
    #Get all integers stored in the tree.
    all_ints = lambda self: self.__root.all_ints()
    #Check whether integer {x} is in the vEB tree.
    tree_member = lambda self, x: False if (x < 0) or (x > self.__max_int) else self.__root.member(x)
    
    #Insert an integer {x} into the rs-vEB tree.
    def tree_insert(self, x):
        
        #Make sure x is within the range [0, 2^{lgu} - 1].
        assert (x >= 0) and (x <= self.__max_int), "!!rs_vEB_tree tree_insert error: x is not within [0, %d]!!" % self.__max_int
        #There is no need to insert {x} again if it is in the tree already.
        if self.__root.member(x): return False
        #Otherwise insert {x} into the rs-vEB tree structure pointed by the root.
        self.__root.insert(x)
        #Update the element count.
        self.__ct += 1
        #The insertion is successfully completed.
        return True
        #func-end
        
    #Delete an integer {x} from the rs-vEB tree.
    def tree_delete(self, x):
        
        #Make sure x is within the range [0, 2^{lgu} - 1].
        assert (x >= 0) and (x <= self.__max_int), "!!rs_vEB_tree tree_delete error: x is not within [0, %d]!!" % self.__max_int
        #The program does nothing if {x} isn't in the tree.
        if not self.__root.member(x): return False
        #Otherwise delete {x} from the rs-vEB tree structure pointed by the root.
        self.__root.delete(x)
        #Update the element count.
        self.__ct -= 1
        #The deletion is successfully completed.
        return True
        #func-end

    #Find the predecessor of {x} in the rs-vEB tree.
    def tree_pred(self, x):
        
        #Return null if the rs-vEB tree is empty or {x} is no greater than 0.
        if (0 == self.__ct) or (x <= 0): return None
        #Return the maximum element if {x} is greater than the upper bound.
        if x > self.__max_int: return self.__root.max()
        #Find the predecessor of {x} if it is within the range [1, 2^{lgu} - 1].
        return self.__root.predecessor(x)
        #func-end
    
    #Find the successor of {x} in the rs-vEB tree.
    def tree_succ(self, x):
        
        #Return null if the rs-vEB tree is empty or {x} is no less than the upper bound.
        if (0 == self.__ct) or (x >= self.__max_int): return None
        #Return the minimum element if {x} is smaller than 0.
        if x < 0: return self.__root.min()
        #Find the sucessor of {x} if it is within the range [0, 2^{lgu} - 2].
        return self.__root.successor(x)
        #func-end
    #class-end


# In[ ]:




