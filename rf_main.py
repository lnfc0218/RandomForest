from scipy.stats import bernoulli
import dtreeclasses
import math
import random
import monkdata as m
import projmain as parser
import itertools


# Reading data
# data = m.monk1  # Example
# attributes = m.attributes
# data = parser.get_bc_dataset(parser.bc_path)
# attributes = parser.bc_attributes
# data = parser.get_hv_dataset(parser.housevotes_path)
# attributes = parser.hv_attributes
# data = parser.get_glass_dataset(parser.glass_path)
# attributes = parser.glass_attributes
data = parser.get_vehicles_dataset(parser.vehicles_path)
attributes = parser.vehicles_attributes
dtreeclasses.set_categ_flag(False)  # Flag for categorical (or non-categorical) inputs)

partition_percentage = 0.66  # Partition percentage for bootstrap replicas (1/3 left out)
Number_replicas = 100
N_times = 100  # Number of times for averaging

M = len(attributes)
list_classes = [x.positive for x in data]  # x.positive indicates class
list_classes = list(set(list_classes))
N_classes = len(set(list_classes))
print("Number of classes:", N_classes)
# Classes must be indicated as follows: [0,1,.., N_classes-1]


def main():
    OOBSingle = 0  # Accumulator for Out-of-bag error (single input splitting: F=1 )
    OOBMultiple = 0  # Accumulator for Out-of-bag error (multiple input splitting: F=log(M)+1 )
    testE = 0  # Accumulator for Test set error (only for single input splitting)
    OOBIndividualSingle = 0  # Accumulator for Out-of-bag indiv. tree error (single input splitting: F=1 )
    OOBIndividualMultiple = 0  # Accumulator for Out-of-bag indiv. tree error (multiple input splitting: F=log(M)+1 )

    F = math.floor(dtreeclasses.log2(M)+1)  # F: Size of group (number of attributes we are using in each node)

    for iteration in range(N_times):  # Loop for averaging errors
        print("Iteration", iteration)

        training_set, test_set = split_data(data)  # Split data between training and test set (90%-10%)

        # Build trees with bagging and random features (Build forest)
        replicas = []  # Bootstrap replicas
        indeces_points = []  # Indeces for out-of-bag estimates
        data_treesSingle = []  # Trees with simple splitting (F==1)
        data_treesMultiple = []  # Trees with multiple splitting (F > 1)

        for i in range(Number_replicas):
            replica, indeces_point = bootstrapReplica(training_set, partition_percentage)
            treeSingle = dtreeclasses.buildTreeMultiple(replica, attributes,list_classes, F=1)
            treeMultiple = dtreeclasses.buildTreeMultiple(replica, attributes, list_classes, F)
            # dtreeclasses.buildTreeMultiple(dataset, attributes, F(split number), depth (default = 1000000))

            replicas.append(replica)
            indeces_points.append(indeces_point)
            data_treesSingle.append(treeSingle)
            data_treesMultiple.append(treeMultiple)

        # Error calculations
        OOBSingle += outOfBagError(training_set, data_treesSingle, indeces_points)
        OOBMultiple += outOfBagError(training_set, data_treesMultiple, indeces_points)
        testE += testSetError(test_set, data_treesSingle)
        OOBIndividualSingle += outOfBagIndividual(training_set, data_treesSingle, indeces_points)
        OOBIndividualMultiple += outOfBagIndividual(training_set, data_treesMultiple, indeces_points)

    # Printing Errors
    OOB = min(OOBSingle, OOBMultiple)  # Lowest OOB error from two groups of inputs
    print("Averaged lowest Out-of-bag error from two groups:", OOB/N_times)
    print("Averaged test set error:", testE/N_times)
    OOBIndividual = min(OOBIndividualSingle, OOBIndividualMultiple)  # Lowest OOB indiv. tree error from two groups of inputs
    print("Averaged lowest Out-of-bag individual tree error from two groups", OOBIndividual/N_times)


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


def testSetError(test_set, data_trees):
    """Computes test set error over whole forest"""
    accum = 0
    for position, sample in enumerate(test_set):
        votes = [0] * N_classes
        # votes is a list with the number of votes per class
        for tree in data_trees:
            votes[int(dtreeclasses.classify(tree, sample))] += 1  # classifyMultiple() returns classification of sample using tree
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
                votes[int(dtreeclasses.classify(tree, sample))] += 1  # returns classification of sample using tree
        if votes.index(max(votes)) == int(sample.positive): # Majority vote =? class
            accum += 1
    return 1 - (accum / len(training_set))

if __name__ == "__main__":
    main()


