"""
@modified by: Diego Yus
"""

import math
import random
import itertools

categ = True
def set_categ_flag(boolean_value):
    """Set categorical variables flag to True or False"""
    global categ
    categ = boolean_value

# def entropy(dataset):
#     "Calculate the entropy of a dataset"
#     n = len(dataset)
#     nPos = len([x for x in dataset if x.positive])
#     nNeg = n - nPos
#     if nPos == 0 or nNeg == 0:
#         return 0.0
#     return -float(nPos)/n * log2(float(nPos)/n) + \
#         -float(nNeg)/n * log2(float(nNeg)/n)

def entropy(dataset, classes):
    "Calculate the entropy of a dataset for several classes"
    n = len(dataset)
    nSamples = []
    for clas in classes:
        nSamples.append(len([x for x in dataset if x.positive == clas]))
    totalEntropy = 0
    for number in nSamples:
        if number == 0:
            continue
        totalEntropy += -float(number)/n * log2(float(number)/n)
    return totalEntropy


def averageGain(dataset, attribute, classes):
    "Calculate the expected information gain when an attribute becomes known"
    weighted = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        weighted += entropy(subset, classes) * len(subset)
    return entropy(dataset, classes) - weighted/len(dataset)


def log2(x):
    "Logarithm, base 2"
    return math.log(x, 2)


def select(dataset, attribute, value):
    "Return subset of data samples where the attribute has the given value"
    if categ == True:
        return [x for x in dataset if x.attribute[attribute] == value]
    else:
        return [x for x in dataset if x.attribute[attribute] >= value and x.attribute[attribute] <= (value + attribute.interval)]

def selectMultiple(dataset, attribute, value):
    "Return subset of data samples where the attributes have the given values (multiple splitting at nodes)"
    subset = []
    for x in dataset:
        flag = True
        for i in range(len(value)):
            if x.attribute[attribute[i]] != value[i]:
                flag = False
        if flag == True:
            subset.append(x)
        # if x.attribute[attribute[0]] == value[0] and x.attribute[attribute[1]] == value[1]:
        #     subset.append(x)
    return subset


def bestAttribute(dataset, attributes, classes):
    "Attribute with highest expected information gain"
    gains = [(averageGain(dataset, a, classes), a) for a in attributes]
    return max(gains, key=lambda x: x[0])[1]
    # Compare elements of list "gains" based on its first index value and return second index value of selected element


def allPositive(dataset):
    "Check if all samples are positive"
    return all([x.positive for x in dataset])


def allNegative(dataset):
    "Check if all samples are negative"
    return not any([x.positive for x in dataset])


def allSameClass(dataset, clas):
    """Checks if all samples of same class"""
    return all([x.positive == clas for x in dataset])


def mostCommon(dataset):
    "Majority class of the dataset"
    pCount = len([x for x in dataset if x.positive])
    nCount = len([x for x in dataset if not x.positive])
    return pCount > nCount


def mostCommon(dataset, classes):
    "Majority class of the dataset for several classes"
    number = []
    for clas in classes:
        number.append(len([x for x in dataset if x.positive == clas]))
    return number.index(max(number))


class TreeNode:
    "Decision tree representation"

    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        "Produce readable (string) representation of the tree"
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'


class TreeLeaf:
    "Decision tree representation for leaf nodes"

    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        "Produce readable (string) representation of this leaf"
        if self.cvalue:
            return '+'
        return '-'


def buildTree(dataset, attributes, maxdepth=1000000):
    "Recursively build a decision tree"

    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        if allPositive(dataset):
            return TreeLeaf(True)
        if allNegative(dataset):
            return TreeLeaf(False)
        return buildTree(dataset, attributes, maxdepth-1)

    default = mostCommon(dataset)
    if maxdepth < 1:
        return TreeLeaf(default)

    # a = bestAttribute(dataset, attributes)  # Select best attribute to split
    index = random.randint(0, len(attributes)-1)  # Select one attribute at random to split
    a = attributes[index]
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft))
                for v in a.values]
    return TreeNode(a, dict(branches), default)

def buildTreeMultiple(dataset, attributes, classes, F, maxdepth=1000000):
    "Recursively build a decision tree (multiple splitting at nodes)"

    # def buildBranch(dataset, default, attributes):
    #     if not dataset:
    #         return TreeLeaf(default)
    #     if allPositive(dataset):
    #         return TreeLeaf(True)
    #     if allNegative(dataset):
    #         return TreeLeaf(False)
    #     if not attributes:
    #         return TreeLeaf(default)
    #     return buildTreeMultiple(dataset, attributes, F, maxdepth-1)

    def buildBranch(dataset, default, attributes, classes):
        if not dataset:
            return TreeLeaf(default)
        for clas in classes:
            if allSameClass(dataset, clas):
                return TreeLeaf(clas)
        if not attributes:
            return TreeLeaf(default)
        return buildTreeMultiple(dataset, attributes, classes, F, maxdepth-1)

    default = mostCommon(dataset, classes)
    if maxdepth < 1:
        return TreeLeaf(default)
    # a = bestAttribute(dataset, attributes)  # Select best attribute to split

    if F == 1:
        index = random.randint(0, len(attributes) - 1)  # Select one attribute at random to split
        a = attributes[index]

    else: # F > 1
        indeces = random.sample(range(0, len(attributes)), min(F, len(attributes)))  # Select F inputs to split from attributes
        # random.sample(range, number of samples): Sampling without replacement (unique samples)
        attributes_selection = list(attributes[i] for i in indeces)  # Select attributes from these inputs
        a = bestAttribute(dataset, attributes_selection, classes)

    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft, classes))
                for v in a.values]

    return TreeNode(a, dict(branches), default)

def classify(tree, sample):
    "Classify a sample using the given decision tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue

    if categ == False:  # Code to be able to use the float numbers of tree.attribute.values as keys for dict.
        i = len(tree.attribute.values)-1
        attri_value = tree.attribute.values[i]  # Greater value of list
        while sample.attribute[tree.attribute] < attri_value:  # Because we always round to the smaller value
            i -= 1
            attri_value = tree.attribute.values[i]
        return classify(tree.branches[attri_value], sample)

    else:
        return classify(tree.branches[sample.attribute[tree.attribute]], sample)


def classifyMultiple(tree, sample, F):
    "Classify a sample using the given decision tree (multiple splitting at nodes)"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    if F == 1:
        return classifyMultiple(tree.branches[sample.attribute[tree.attribute]], sample, F)
    else:
        # return classifyMultiple(tree.branches[(sample.attribute[tree.attribute[0]], sample.attribute[tree.attribute[1]])], sample)
        tupla = tuple(sample.attribute[tree.attribute[i]] for i in range(len(tree.attribute)))
        return classifyMultiple(tree.branches[tupla], sample, F)


def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)


def allPruned(tree):
    "Return a list if trees, each with one node replaced by the corresponding default class"
    if isinstance(tree, TreeLeaf):
        return ()
    alternatives = (TreeLeaf(tree.default),)
    for v in tree.branches:
        for r in allPruned(tree.branches[v]):
            b = tree.branches.copy()
            b[v] = r
            alternatives += (TreeNode(tree.attribute, b, tree.default),)
    return alternatives
