"""
@author: Diego Yus
"""

from scipy.stats import bernoulli
import dtreeclasses
import math
import random
import monkdata as m
import projmain as parser
import itertools
import matplotlib.pyplot as plt
import numpy as np

# Reading data
# data = m.monk1  # Example
# attributes = m.attributes
# data = parser.get_hv_dataset(parser.housevotes_path)
# attributes = parser.hv_attributes
# dtreeclasses.set_categ_flag(True)
# data = parser.get_diabetes_dataset(parser.diabetes_path)
# attributes = parser.diabetes_attributes
data = parser.get_sonar_dataset(parser.sonar_path)
attributes = parser.sonar_attributes
dtreeclasses.set_categ_flag(False)  # Flag for categorical (or non-categorical) inputs)

partition_percentage = 0.66  # Partition percentage for bootstrap replicas (1/3 left out)
Number_replicas = 100  # Number of trees
N_times = 80  # Number of times for averaging

M = len(attributes)  # Number of attributes
list_classes = [x.positive for x in data]  # x.positive indicates class
list_classes = list(set(list_classes))
N_classes = len(list_classes)
print("Number of classes:", N_classes)
# Classes must be indicated as follows: [0,1,.., N_classes-1]



def main():

    OOBList = []
    testEList =[]
    # OOBIndividualList = []
    strengthList = []
    correlationList = []

    # F = math.floor(dtreeclasses.log2(M)+1)  # F: Size of group (number of attributes we are using in each node)
    F = 51
    for f in range(1, F):
        OOBSingle = 0  # Accumulator for Out-of-bag error (single input splitting: F=1 )
        OOBMultiple = 0  # Accumulator for Out-of-bag error (multiple input splitting: F=log(M)+1 )
        testE = 0  # Accumulator for Test set error (only for single input splitting)
        OOBIndividualSingle = 0  # Accumulator for Out-of-bag indiv. tree error (single input splitting: F=1 )
        OOBIndividualMultiple = 0  # Accumulator for Out-of-bag indiv. tree error (multiple input splitting: F=log(M)+1 )
        strength = 0
        correlation = 0

        for iteration in range(N_times):  # Loop for averaging errors
            print("Iteration", iteration)

            training_set, test_set = split_data(data)  # Split data between training and test set (90%-10%)

            # Build trees with bagging and random features (Build forest)
            replicas = []  # Bootstrap replicas
            indeces_points = []  # Indeces for out-of-bag estimates
            # data_treesSingle = []  # Trees with simple splitting (F==1)
            data_treesMultiple = []  # Trees with multiple splitting (F > 1)

            # for i in range(Number_replicas):
            for i in range(Number_replicas):
                replica, indeces_point = bootstrapReplica(training_set, partition_percentage)
                #treeSingle = dtreeclasses.buildTreeMultiple(replica, attributes,list_classes, F=1)
                treeMultiple = dtreeclasses.buildTreeMultiple(replica, attributes, list_classes, f)
                # dtreeclasses.buildTreeMultiple(dataset, attributes, F(split number), depth (default = 1000000))

                replicas.append(replica)
                indeces_points.append(indeces_point)
                # data_treesSingle.append(treeSingle)
                data_treesMultiple.append(treeMultiple)

            # Error calculations
            # OOBSingle += outOfBagError(training_set, data_treesSingle, indeces_points)
            OOBMultiple += outOfBagError(training_set, data_treesMultiple, indeces_points)
            testE += testSetError(test_set, data_treesMultiple)
            # OOBIndividualSingle += outOfBagIndividual(training_set, data_treesSingle, indeces_points)
            # OOBIndividualMultiple += outOfBagIndividual(training_set, data_treesMultiple, indeces_points)
            strength += strengthComp(training_set, data_treesMultiple, indeces_points)
            correlation += correlationComp(training_set, data_treesMultiple, indeces_points)

        # Printing Errors
        # OOB = min(OOBSingle, OOBMultiple)/N_times  # Lowest OOB error from two groups of inputs
        OOB = OOBMultiple/N_times
        testE = testE/N_times
        strength = strength/N_times
        correlation = correlation/N_times
        # OOBIndividual = min(OOBIndividualSingle,
        #                     OOBIndividualMultiple)/N_times  # Lowest OOB indiv. tree error from two groups of inputs
        print("Averaged lowest Out-of-bag error from two groups:", OOB)
        print("Averaged test set error:", testE)
        print("Strength:", strength)
        print("Correlation:", correlation)
        # print("Averaged lowest Out-of-bag individual tree error from two groups", OOBIndividual)

        OOBList.append(OOB)
        testEList.append(testE)
        strengthList.append(strength)
        correlationList.append(correlation)

        np.save('OOBErrors2.npy', np.array(OOBList))
        np.save('TestE2.npy', np.array(testEList))
        np.save('Strength.npy', np.array(strengthList))
        np.save('Correlation.npy', np.array(correlationList))
        # OOBIndividualList.append(OOBIndividual)

    fig= plt.figure()
    ax = plt.gca()
    # plt.plot(range(1,F), OOBList, marker='o', linestyle='--', label='Out-of-bag error' )
    # plt.plot(range(1,F), testEList, marker='o', linestyle='--', label='Test Error')
    plt.plot(range(1, F), strengthList, marker='o', linestyle='--', label='Strength')
    plt.plot(range(1, F), correlationList, marker='o', linestyle='--', label='Correlation')
    # plt.plot(Number_replicas, OOBIndividualList, marker='o', linestyle='--', label='Out-of-bag error individual')
    plt.xlabel('F')
    plt.ylabel('Y Variables')
    plt.title('Correlation and strength')
    legend = ax.legend()
    plt.show()


def outOfBagIndividual(training_set, data_trees, indeces_points):
    """Computes out-of-bag error for individual tree (averaged over all trees)"""
    accum = 0  # accumulator for averaged error of all individual tree
    for i, tree in enumerate(data_trees):
        accum1 = 0  # accumulator for error of a single tree
        count = 0  # counter for number of samples out-of-bag of single tree
        for position, sample in enumerate(training_set):
            if indeces_points[i][position] == False:  # If the sample does not belong to the replica/tree
                accum1 += (int(dtreeclasses.classify(tree, sample)) == int(sample.positive))
                # If sample well classified, +1 in accumulator; otherwise +0. // sample.positive indicates class
                count += 1
        accum += accum1/count  # Add the error of each single tree over all the out-of-bag samples to total error
    return 1 - (accum/len(data_trees))

def correlationComp(training_set, data_trees, indeces_points):
    expected_value = []
    std = []
    for i, tree in enumerate(data_trees):
        values = []
        for position, sample in enumerate(training_set):
            if indeces_points[i][position] == False:
                if int(dtreeclasses.classify(tree, sample)) == int(sample.positive):
                    values.append(1)
                else:
                    values.append(-1)
        expected_value.append(np.mean(np.array(values)))
        std.append(np.std(np.array(values)))

    correlation = []
    for i, tree in enumerate(data_trees):
        for j in range(i+1, len(data_trees)):
            values = []
            for position, sample in enumerate(training_set):
                if indeces_points[i][position] == False and indeces_points[j][position] == False:
                    if int(dtreeclasses.classify(tree, sample)) == int(dtreeclasses.classify(data_trees[j], sample)):
                        values.append(1)
                    else:
                        values.append(-1)
            cov = np.mean(np.array(values))-expected_value[i]*expected_value[j]
            correlation.append(cov/(std[i]*std[j]))

    cor = np.mean(np.array(correlation))
    return cor



def strengthComp(training_set, data_trees, indeces_points):
    """Computes test set error over whole forest"""
    accum = 0
    for position, sample in enumerate(training_set):
        votes_correct = 0  # Number if votes if class is correct
        votes_incorrect = 0  # Number of votes if class in incorrect
        count = 0
        for i, tree in enumerate(data_trees):
            if indeces_points[i][position] == False:
                if int(dtreeclasses.classify(tree, sample)) == int(sample.positive):
                    votes_correct += 1
                else:
                    votes_incorrect += 1
            count += 1
        # print("Correct votes:", votes_correct)
        # print("Incorrect votes", votes_incorrect)
        accum += (votes_correct - votes_incorrect)/count
    return accum/len(training_set)


def testSetError(test_set, data_trees):
    """Computes test set error over whole forest"""
    accum = 0
    for position, sample in enumerate(test_set):
        votes = [0] * N_classes
        # votes is a list with the number of votes per class
        for tree in data_trees:
            # votes[dtreeclasses.classifyMultiple(tree, sample, F=1)] += 1  # classifyMultiple() returns classification of sample using tree
            votes[int(dtreeclasses.classify(tree, sample))] += 1
        if votes.index(max(votes)) == int(sample.positive):  # Again, sample.positive indicates class
            accum += 1
    return 1 - (accum / len(test_set))


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
    training_set = tuple(dataset[:math.floor(N_data*0.9)])  # Returned as tuple for proper running
    test_set = tuple(dataset[math.floor(N_data*0.9):])      # Returned as tuple for proper running
    return training_set, test_set


def outOfBagError(training_set, data_trees, indeces_points):
    """Computes out-of-bag error for forest"""
    accum = 0
    for position, sample in enumerate(training_set):
        votes = [0] * N_classes
        # votes is a list with the number of votes per class
        for i, tree in enumerate(data_trees):
            if indeces_points[i][position] == False:  # point does not belongs to replica[i]
                # votes[dtreeclasses.classifyMultiple(tree, sample, F)] += 1  # returns classification of sample using tree
                votes[int(dtreeclasses.classify(tree, sample))] += 1
        if votes.index(max(votes)) == int(sample.positive): # Majority vote =? class
            accum += 1
    return 1 - (accum / len(training_set))

if __name__ == "__main__":
    main()


