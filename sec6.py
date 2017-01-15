#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 16:41:13 2017

@author: pangh
"""

from scipy.stats import bernoulli
import dtreeclasses
import math
import random
import monkdata as m
import projmain as parser
import itertools
import numpy as np
import matplotlib.pyplot as plt


# Reading data
#data = parser.get_bc_dataset(parser.bc_path)
#attributes = parser.bc_attributes
data = parser.get_sonar_dataset(parser.sonar_path)
attributes = parser.sonar_attributes
dtreeclasses.set_categ_flag(False)  # Flag for categorical (or non-categorical) inputs)
#dtreeclasses.set_categ_flag(True) 

partition_percentage = 0.66  # Partition percentage for bootstrap replicas (1/3 left out)
Number_replicas = 100
#N_times = 100  # Number of times for averaging
N_times = 1

M = len(attributes)
list_classes = [x.positive for x in data]  # x.positive indicates class
list_classes = list(set(list_classes))
N_classes = len(set(list_classes))
print("Number of classes:", N_classes)
# Classes must be indicated as follows: [0,1,.., N_classes-1]

def bootstrapReplica(train_set, percentage):
    """Creates bootstrap replicas"""
    N_data = len(train_set)
    indeces = bernoulli.rvs(percentage, size=N_data)  # Vector of 0's (1 - percentage) and 1's (percentage)
    replica = [val for is_1, val in zip(indeces, train_set) if is_1]  # Select samples from train_set where indeces[i] = 1
    return replica, indeces

def split_data(dataset):
    """Splits dataset in training set and test set"""
    dataset = list(dataset)
    random.shuffle(dataset)  # Shuffle dataset for random split
    dataset = tuple(dataset)

    N_data = len(dataset)
    training_set = tuple(dataset[:int(math.floor(N_data*0.9))])  # Returned as tuple for proper running
    test_set = tuple(dataset[int(math.floor(N_data*0.9)):])      # Returned as tuple for proper running
    return training_set, test_set

def margin(set, data_trees):    
    """Computes margin function to test if mr(X,Y)==E(Theta,X,Y) w.r.t Theta;
    They are equal just as the paper says, so either margin() or rawMargin() is fine to get strength"""
    accum = 0
    mr=[]
    for position, sample in enumerate(set):
        votes = [0] * N_classes
        for tree in data_trees:
            votes[int(dtreeclasses.classify(tree, sample))] += 1  
#        print(votes)
        ind=int(sample.positive)
        margin=(votes[ind]-max(votes[:ind]+votes[ind+1:]))/len(data_trees)
        mr.append(margin)
        if votes.index(max(votes)) == ind: 
            accum += 1
#    print('margin function = '+str(mr))
    strength = sum(mr)/len(set)
#    print('strength = '+str(strength))
    return strength
    
def rawMargin(set, data_trees):
    """Computes raw margin function;
    TODO: fix conditions for more than two classes."""
    rmg=[]
    for tree in data_trees:
        vec=[]
        for position, sample in enumerate(set):
            ind_true=int(sample.positive)
            ind_esti=int(dtreeclasses.classify(tree, sample))
            # for binary classification, naive conditions(for sat data with 6 classes, need to merge with margin())
            if ind_true == ind_esti:
                vec.append(1)
            else:
                vec.append(-1)
#        print('vec='+str(vec))
        rmg.append(vec)
#    print('length of rmg='+str(len(rmg)))
    print(np.sum(rmg)/len(set)/Number_replicas)# if it equals strength, then we are doing right
    return rmg

    
def getCorr(rmg):
    corr_rmg=np.corrcoef(rmg)
    return np.nanmean(corr_rmg)

whole_strength=[]
whole_correlation=[]

#for F in range(5):
for F in range(50):
    F+=1
    print('-------------------------')
    print('Group size F = '+str(F))
    
    row_strength=[]
    row_correlation=[]

#    for iter in range(8):
    for iter in range(80):
#        print('Group size F = '+str(F))
        print('Iteration = '+str(iter))
        training_set, test_set = split_data(data)  # Split data between training and test set (90%-10%)
        # Build trees with bagging and random features (Build forest)
        replicas = []  # Bootstrap replicas
        indeces_points = []  # Indeces for out-of-bag estimates
        data_trees = []  # initialize trees
        for i in range(Number_replicas):
            replica, indeces_point = bootstrapReplica(training_set, partition_percentage)
            treeSingle = dtreeclasses.buildTreeMultiple(replica, attributes,list_classes, F)
            replicas.append(replica)
            indeces_points.append(indeces_point)
            data_trees.append(treeSingle)
        # Error calculations
        strength = margin(test_set,data_trees)
        rmg = rawMargin(test_set,data_trees)
        correlation = getCorr(rmg)
#        print('strength = '+str(strength))
#        print('correlation = '+str(correlation))
        row_strength.append(strength)
        row_correlation.append(correlation)
    print('Average strength = '+str(np.mean(row_strength)))
    print('Average correlation = '+str(np.mean(row_correlation)))
    whole_strength.append(row_strength)
    whole_correlation.append(row_correlation)
    

avg_strength=np.mean(whole_strength,axis=1)
avg_correlation=np.mean(whole_correlation,axis=1)

F=np.arange(1,51)

plt.figure()
line_strength=plt.plot(F,avg_strength, color='blue',marker='o' , linestyle='--',label='strength')
line_correlation=plt.plot(F,avg_correlation, color='green',marker='o' , linestyle='--',label='correlation')
plt.xlim([0,50])
plt.ylim([0,.6])
plt.xlabel('$F$')
plt.legend()
plt.title('Strength and Correlation')
plt.show()
