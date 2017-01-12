import pandas as p
import numpy

glass_path = "Datasets/glass.data"
bc_path = "Datasets/breast-cancer-wisconsin-changed.data"
housevotes_path = "Datasets/house-votes-84-changed.data"

p.set_option('display.max_row', 1000)


class GlassSample:
    "Representation of data samples"

    def __init__(self, sample_class, values, identity):
        # self.sample_class = sample_class
        self.positive = sample_class  # Changed for easiness. When multiple classes implemented, change back
        self.attribute = dict(zip(glass_attributes, values))
        self.identity = identity

class BCSample:
    "Representation of data samples"

    def __init__(self, sample_class, values, identity):
        self.positive = sample_class  # Changed name so we don't have to change whole code
        self.attribute = dict(zip(bc_attributes, values))
        self.identity = identity

class HouseVotesSample:
    "Represenation of data samles"

    def __init__(self, class_name, values):
        self.positive = class_name
        self.attribute = dict(zip(hv_attributes, values))


class Attribute:
    "Label for each attribute"

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __repr__(self):
        return self.name

class AttributeC:
    "Label for each attribute when continuous inputs"

    def __init__(self, name, values, interval):
        self.name = name
        self.values = values
        self.interval = interval

    def __repr__(self):
        return self.name


# glass_attributes = (Attribute('A1', tuple(numpy.linspace(1.5112,1.5339).tolist())),
#               Attribute('A2', tuple(numpy.linspace(10.73,17.38).tolist())),
#               Attribute('A3', tuple(numpy.linspace(0 , 4.49).tolist())),
#               Attribute('A4', tuple(numpy.linspace(0.29, 3.5).tolist())),
#               Attribute('A5', tuple(numpy.linspace(69.81, 75.41).tolist())),
#               Attribute('A6', tuple(numpy.linspace(0, 6.21).tolist())),
#               Attribute('A7', tuple(numpy.linspace(5.43, 16.19).tolist())),
#               Attribute('A8', tuple(numpy.linspace(0, 3.15).tolist())),
#               Attribute('A9', tuple(numpy.linspace(0, 0.51).tolist())))

N = 15  # Number of bins per attribute

# For each attribute, we make N bins between min and max values.
# bin = [min_Value, min_Value + interval]
# To do so, we use np.linspace and delete last element
# interval computed as (max_Value - min_Value) / N (see below)
a1 = numpy.linspace(1.5112, 1.5339, (N+1)).tolist()[:-1]
a2 = numpy.linspace(10.73, 17.38, (N+1)).tolist()[:-1]
a3 = numpy.linspace(0, 4.49, (N+1)).tolist()[:-1]
a4 = numpy.linspace(0.29, 3.5, (N+1)).tolist()[:-1]
a5 = numpy.linspace(69.81, 75.41, (N+1)).tolist()[:-1]
a6 = numpy.linspace(0, 6.21, (N+1)).tolist()[:-1]
a7 = numpy.linspace(5.43, 16.19, (N+1)).tolist()[:-1]
a8 = numpy.linspace(0, 3.15,(N+1)).tolist()[:-1]
a9 = numpy.linspace(0, 0.51,(N+1)).tolist()[:-1]

glass_attributes = (AttributeC('A1', tuple(a1), (1.5339-1.5112)/N),
              AttributeC('A2', tuple(a2), (17.38-10.73)/N),
              AttributeC('A3', tuple(a3), (4.49-0)/N),
              AttributeC('A4', tuple(a4), (3.5-0.29)/N),
              AttributeC('A5', tuple(a5), (75.41-69.81)/N),
              AttributeC('A6', tuple(a6), (6.21-0)/N),
              AttributeC('A7', tuple(a7), (16.19-5.43)/N),
              AttributeC('A8', tuple(a8), (3.15-0)/N),
              AttributeC('A9', tuple(a9), (0.51-0)/N))


# bc_attributes = (Attribute('A1', tuple(numpy.linspace(3,4).tolist())),
#               Attribute('A2', tuple(numpy.linspace(3,2).tolist())),
#               Attribute('A3', tuple(numpy.linspace(2,3).tolist())),
#               Attribute('A4', tuple(numpy.linspace(2,1).tolist())),
#               Attribute('A5', tuple(numpy.linspace(3,2).tolist())),
#               Attribute('A6', tuple(numpy.linspace(3, 3).tolist())),
#               Attribute('A7', tuple(numpy.linspace(5, 16).tolist())),
#               Attribute('A8', tuple(numpy.linspace(0, 3).tolist())))

#  Don't know reason why set these previous values for attributes domains
bc_attributes = (Attribute('A1', tuple(range(1,11))),
              Attribute('A2', tuple(range(1,11))),
              Attribute('A3', tuple(range(1,11))),
              Attribute('A4', tuple(range(1,11))),
              Attribute('A5', tuple(range(1,11))),
              Attribute('A6', tuple(range(1,11))),
              Attribute('A7', tuple(range(1,11))),
              Attribute('A8', tuple(range(1,11))),
              Attribute('A9', tuple(range(1,11))))

# hv_attributes = (Attribute('A1', tuple(numpy.linspace(3,4).tolist())),
#               Attribute('A2', tuple(numpy.linspace(3,2).tolist())),
#               Attribute('A3', tuple(numpy.linspace(2,3).tolist())),
#               Attribute('A4', tuple(numpy.linspace(2,1).tolist())),
#               Attribute('A5', tuple(numpy.linspace(3,2).tolist())),
#               Attribute('A6', tuple(numpy.linspace(3, 3).tolist())),
#               Attribute('A7', tuple(numpy.linspace(5, 16).tolist())),
#               Attribute('A8', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A9', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A10', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A11', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A12', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A13', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A14', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A15', tuple(numpy.linspace(0, 3).tolist())),
#               Attribute('A16', tuple(numpy.linspace(0, 3).tolist())))

#  Don't know reason why set these previous values for attributes domains
hv_attributes = (Attribute('A1', (2, 3, 4)),
              Attribute('A2', (2, 3, 4)),
              Attribute('A3', (2, 3, 4)),
              Attribute('A4', (2, 3, 4)),
              Attribute('A5', (2, 3, 4)),
              Attribute('A6', (2, 3, 4)),
              Attribute('A7', (2, 3, 4)),
              Attribute('A8', (2, 3, 4)),
              Attribute('A9', (2, 3, 4)),
              Attribute('A10', (2, 3, 4)),
              Attribute('A11', (2, 3, 4)),
              Attribute('A12', (2, 3, 4)),
              Attribute('A13', (2, 3, 4)),
              Attribute('A14', (2, 3, 4)),
              Attribute('A15', (2, 3, 4)),
              Attribute('A16', (2, 3, 4)))


def get_bc_dataset(path):
    '''
    Return a list containing samples as sample_class, (tuple of features), id
    Attribute names taken from glass.name file'''

    dataframe = p.read_csv(path, names = ['id', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape', 'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'])

    # transpose the dataframe so that it is easier to pick them by just selecting a column
    data = dataframe.T

    # list for storing each individual sample as a list
    list_for_samples = []
    for col in range(data.shape[1]):
        list_for_samples.append(data[col].tolist())

    # list for storing samples as sample_class, (tuple of features), id as required by the ml program
    structured_data = []
    for row in list_for_samples:
        structured_data.append([row[-1],tuple(row[1:len(row)-1]),row[0]])

    # iterate through items in sampels and return the Sample()
   # print structured_data
    samples = ()
    for item in structured_data:
      if item[0] == 2:
            item[0] = 0
      if item[0] == 4:
            item[0] = 1
      samples += (BCSample(item[0], item[1], item[2]),)

    return samples

def get_glass_dataset(path):
    '''
    Return a list containing samples as sample_class, (tuple of features), id
    Attribute names taken from glass.name file'''

    dataframe = p.read_csv(path, names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'])

    # transpose the dataframe so that it is easier to pick them by just selecting a column
    data = dataframe.T

    # list for storing each individual sample as a list
    list_for_samples = []
    for col in range(data.shape[1]):
        list_for_samples.append(data[col].tolist())

    # list for storing samples as sample_class, (tuple of features), id as required by the ml program
    structured_data = []
    for row in list_for_samples:
        structured_data.append([row[-1],tuple(row[1:len(row)-1]),row[0]])

    # iterate through items in sampels and return the Sample()
    samples = ()
    for item in structured_data[:130]:  # So that we only have two classes and is testable with the code
        samples += (GlassSample(item[0]-1, item[1], item[2]),)
    # for item in structured_data:
    #   samples += (GlassSample(item[0], item[1], item[2]),)

    return samples
 
# 0 = rep
# 1 = democ
# 2 = n
# 3 = y
# 4 = ?
def get_hv_dataset(path):
    '''
    Return a list containing samples as sample_class, (tuple of features), id
    Attribute names taken from glass.name file'''

    dataframe = p.read_csv(path, names = ['house','handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'])
    # transpose the dataframe so that it is easier to pick them by just selecting a column
    data = dataframe.T
    # list for storing each individual sample as a list
    list_for_samples = []
    for col in range(data.shape[1]):
      list_for_samples.append(data[col].tolist())

#    print list_for_samples
    # list for storing samples as sample_class, (tuple of features), id as required by the ml program
    structured_data = []
    for row in list_for_samples:
      structured_data.append([row[0],tuple(row[1:len(row)])])

    # iterate through items in sampels and return the Sample()
   # print structured_data
    samples = ()
    for item in structured_data:
      samples += (HouseVotesSample(item[0], item[1]),)

    return samples
 

# Example for glass dataset
# samples = get_glass_dataset(glass_path)
# print len(samples)
# for item in samples:
#    print item.sample_class, item.attribute, item.identity


# # Example for bc dataset
# samples = get_bc_dataset(bc_path)
# for item in samples:
#    print item.sample_class, item.attribute, item.identity
# print len(samples)

# Not working yet? 
# samples = get_hv_dataset(housevotes_path)
# for item in samples:
#   print item.class_name, item.attribute
# print len(samples)