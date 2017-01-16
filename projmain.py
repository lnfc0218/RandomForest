"""
@author: Diego Yus
"""

import pandas as p
import numpy

glass_path = "Datasets/glass.data"
bc_path = "Datasets/breast-cancer-wisconsin-changed.data"
housevotes_path = "Datasets/house-votes-84-changed.data"
sonar_path = "Datasets/sonar-changed.data"
diabetes_path = "Datasets/pima-indians-diabetes.data"
vehicles_path = "Datasets/vehicle-changed.txt"

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

class SonarSample:
    "Representation of data samples"

    def __init__(self, class_name, values):
        self.positive = class_name
        self.attribute = dict(zip(sonar_attributes, values))

class DiabetesSample:
    "Representation of data samples"

    def __init__(self, class_name, values):
        self.positive = class_name
        self.attribute = dict(zip(diabetes_attributes, values))

class VehiclesSample:
    "Representation of data samples"

    def __init__(self, class_name, values):
        self.positive = class_name
        self.attribute = dict(zip(vehicles_attributes, values))


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


N=2
vehicles_attributes = (AttributeC('A1', tuple(numpy.linspace(73,119,N+1).tolist()[:-1]), (119-73)/N),
AttributeC('A2', tuple(numpy.linspace(33,59,N+1).tolist()[:-1]), (59-33)/N),
AttributeC('A3', tuple(numpy.linspace(40,112,N+1).tolist()[:-1]), (112-40)/N),
AttributeC('A4', tuple(numpy.linspace(104,333,N+1).tolist()[:-1]), (333-104)/N),
AttributeC('A5', tuple(numpy.linspace(47,138,N+1).tolist()[:-1]), (138-47)/N),
AttributeC('A6', tuple(numpy.linspace(2,55,N+1).tolist()[:-1]), (55-2)/N),
AttributeC('A7', tuple(numpy.linspace(112,265,N+1).tolist()[:-1]), (265-112)/N),
AttributeC('A8', tuple(numpy.linspace(26,61,N+1).tolist()[:-1]), (61-26)/N),
AttributeC('A9', tuple(numpy.linspace(17,29,N+1).tolist()[:-1]), (29-17)/N),
AttributeC('A10', tuple(numpy.linspace(118,188,N+1).tolist()[:-1]), (188-118)/N),
AttributeC('A11', tuple(numpy.linspace(130,320,N+1).tolist()[:-1]), (320-130)/N),
AttributeC('A12', tuple(numpy.linspace(184,1018,N+1).tolist()[:-1]), (1018-184)/N),
AttributeC('A13', tuple(numpy.linspace(109,268,N+1).tolist()[:-1]), (268-109)/N),
AttributeC('A14', tuple(numpy.linspace(59,135,N+1).tolist()[:-1]), (135-59)/N),
AttributeC('A15', tuple(numpy.linspace(0,22,N+1).tolist()[:-1]), (22-0)/N),
AttributeC('A16', tuple(numpy.linspace(0,41,N+1).tolist()[:-1]), (41-0)/N),
AttributeC('A17', tuple(numpy.linspace(176,206,N+1).tolist()[:-1]), (206-176)/N),
AttributeC('A18', tuple(numpy.linspace(181,211,N+1).tolist()[:-1]), (211-181)/N))


N = 2
d1 = numpy.linspace(0, 17, 18).tolist()[:-1]
d2 = numpy.linspace(0, 199, (N+1)).tolist()[:-1]
d3 = numpy.linspace(0, 122, (N+1)).tolist()[:-1]
d4 = numpy.linspace(0, 99, (N+1)).tolist()[:-1]
d5 = numpy.linspace(0, 846, (N+1)).tolist()[:-1]
d6 = numpy.linspace(0, 67.1, (N+1)).tolist()[:-1]
d7 = numpy.linspace(0, 2.42, (N+1)).tolist()[:-1]
d8 = numpy.linspace(21, 81, (N+1)).tolist()[:-1]

diabetes_attributes = (AttributeC('A1', tuple(d1), 17/18),
              AttributeC('A2', tuple(d2), 199/N),
              AttributeC('A3', tuple(d3), 122/N),
              AttributeC('A4', tuple(d4), 99/N),
              AttributeC('A5', tuple(d5), 846/N),
              AttributeC('A6', tuple(d6), 67.1/N),
              AttributeC('A7', tuple(d7), 2.42/N),
              AttributeC('A8', tuple(d8), (81-21)/N))


N = 2
s1 = numpy.linspace(0.0, 1.0, (N+1)).tolist()[:-1]

sonar_attributes = (AttributeC('A1', tuple(numpy.linspace(0,0.1371,N+1).tolist()[:-1]), 0.1371/N),
AttributeC('A2', tuple(numpy.linspace(0,0.24,N+1).tolist()[:-1]), 0.24/N),
AttributeC('A3', tuple(numpy.linspace(0,0.31,N+1).tolist()[:-1]), 0.31/N),
AttributeC('A4', tuple(numpy.linspace(0,0.45,N+1).tolist()[:-1]), 0.45/N),
AttributeC('A5', tuple(numpy.linspace(0,0.42,N+1).tolist()[:-1]), 0.42/N),
AttributeC('A6', tuple(numpy.linspace(0,0.4,N+1).tolist()[:-1]), 0.4/N),
AttributeC('A7', tuple(numpy.linspace(0,0.4,N+1).tolist()[:-1]), 0.4/N),
AttributeC('A8', tuple(numpy.linspace(0,0.5,N+1).tolist()[:-1]), 0.5/N),
AttributeC('A9', tuple(numpy.linspace(0,0.7,N+1).tolist()[:-1]), 0.7/N),
AttributeC('A10', tuple(numpy.linspace(0,0.72,N+1).tolist()[:-1]), 0.72/N),
AttributeC('A11', tuple(numpy.linspace(0,0.75,N+1).tolist()[:-1]), 0.75/N),
AttributeC('A12', tuple(numpy.linspace(0,0.71,N+1).tolist()[:-1]), 0.71/N),
AttributeC('A13', tuple(numpy.linspace(0,0.72,N+1).tolist()[:-1]), 0.72/N))
for i in range(14,43):
    sonar_attributes += (AttributeC('A'+str(i+1), tuple(s1), 1/N),)
sonar_attributes2 = (AttributeC('A43', tuple(numpy.linspace(0,0.78,N+1).tolist()[:-1]), 0.78/N),
AttributeC('A44', tuple(numpy.linspace(0,0.78,N+1).tolist()[:-1]), 0.78/N),
AttributeC('A45', tuple(numpy.linspace(0,0.78,N+1).tolist()[:-1]), 0.78/N),
AttributeC('A46', tuple(numpy.linspace(0,0.73,N+1).tolist()[:-1]), 0.73/N),
AttributeC('A47', tuple(numpy.linspace(0,0.6,N+1).tolist()[:-1]), 0.6/N),
AttributeC('A48', tuple(numpy.linspace(0,0.35,N+1).tolist()[:-1]), 0.35/N),
AttributeC('A49', tuple(numpy.linspace(0,0.2,N+1).tolist()[:-1]), 0.2/N),
AttributeC('A50', tuple(numpy.linspace(0,0.1,N+1).tolist()[:-1]), 0.1/N),
AttributeC('A51', tuple(numpy.linspace(0,0.11,N+1).tolist()[:-1]), 0.11/N),
AttributeC('A52', tuple(numpy.linspace(0,0.08,N+1).tolist()[:-1]), 0.08/N),
AttributeC('A53', tuple(numpy.linspace(0,0.04,N+1).tolist()[:-1]), 0.04/N),
AttributeC('A54', tuple(numpy.linspace(0,0.04,N+1).tolist()[:-1]), 0.04/N),
AttributeC('A55', tuple(numpy.linspace(0,0.05,N+1).tolist()[:-1]), 0.05/N),
AttributeC('A56', tuple(numpy.linspace(0,0.04,N+1).tolist()[:-1]), 0.04/N),
AttributeC('A57', tuple(numpy.linspace(0,0.04,N+1).tolist()[:-1]), 0.04/N),
AttributeC('A58', tuple(numpy.linspace(0,0.05,N+1).tolist()[:-1]), 0.05/N),
AttributeC('A59', tuple(numpy.linspace(0,0.04,N+1).tolist()[:-1]), 0.04/N),
AttributeC('A60', tuple(numpy.linspace(0,0.05,N+1).tolist()[:-1]), 0.05/N))
sonar_attributes += sonar_attributes2

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


def get_sonar_dataset(path):
    '''
    Return a list containing samples as sample_class, (tuple of features), id
    Attribute names taken from sonar.name file'''

    names1 = ['Fr1', 'Fr2', 'Fr3', 'Fr4', 'Fr5', 'Fr6', 'Fr7', 'Fr8', 'Fr9', 'Fr10']
    names2 = ['Fr11', 'Fr12', 'Fr13', 'Fr14', 'Fr15', 'Fr16', 'Fr17', 'Fr18', 'Fr19', 'Fr20']
    names3 = ['Fr21', 'Fr22', 'Fr23', 'Fr24', 'Fr25', 'Fr26', 'Fr27', 'Fr28', 'Fr29', 'Fr30']
    names4 = ['Fr31', 'Fr32', 'Fr33', 'Fr34', 'Fr35', 'Fr36', 'Fr37', 'Fr38', 'Fr39', 'Fr40']
    names5 = ['Fr41', 'Fr42', 'Fr43', 'Fr44', 'Fr45', 'Fr46', 'Fr47', 'Fr48', 'Fr49', 'Fr50']
    names6 = ['Fr51', 'Fr52', 'Fr53', 'Fr54', 'Fr55', 'Fr56', 'Fr57', 'Fr58', 'Fr59', 'Fr60', 'Type']
    names = names1+names2+names3+names4+names5+names6
    dataframe = p.read_csv(path, names = names)

    # transpose the dataframe so that it is easier to pick them by just selecting a column
    data = dataframe.T

    # list for storing each individual sample as a list
    list_for_samples = []
    for col in range(data.shape[1]):
        list_for_samples.append(data[col].tolist())

    # total_list = []
    # for k in range(60):
    #     new_list = []
    #     for item in list_for_samples:
    #         new_list.append(item[k])
    #     total_list.append(new_list)
    # print(total_list)

    # maximums = [max(x) for x in total_list]
    # minimums = [min(x) for x in total_list]
    # for i, item in enumerate(maximums):
    #     print(i+1, minimums[i], item)


    # list for storing samples as sample_class, (tuple of features), id as required by the ml program
    structured_data = []
    for row in list_for_samples:
        structured_data.append([row[-1],tuple(row[0:len(row)-1])])

    # iterate through items in sampels and return the Sample()
    samples = ()
    for item in structured_data:
      samples += (SonarSample(item[0], item[1]),)

    return samples
 

def get_diabetes_dataset(path):
    '''
    Return a list containing samples as sample_class, (tuple of features), id
    Attribute names taken from pima-indians-diabetes.name file'''

    dataframe = p.read_csv(path, names = ['Times pregnant', 'Plasma glucose concentration', 'Plasma glucose concentration ', 'Triceps skin fold thickness', '2-Hour serum insulin', 'Body mass index', 'Diabetes pedigree function', 'Age', 'Type'])

    # transpose the dataframe so that it is easier to pick them by just selecting a column
    data = dataframe.T

    # list for storing each individual sample as a list
    list_for_samples = []
    for col in range(data.shape[1]):
        list_for_samples.append(data[col].tolist())

    # total_list = []
    # for k in range(8):
    #     new_list = []
    #     for item in list_for_samples:
    #         new_list.append(item[k])
    #     total_list.append(new_list)
    # print(total_list)
    #
    # maximums = [max(x) for x in total_list]
    # minimums = [min(x) for x in total_list]
    # for i, item in enumerate(maximums):
    #     print(i+1, minimums[i], item)

    # list for storing samples as sample_class, (tuple of features), id as required by the ml program
    structured_data = []
    for row in list_for_samples:
        structured_data.append([row[-1],tuple(row[0:len(row)-1])])

    # iterate through items in sampels and return the Sample()
    samples = ()
    for item in structured_data:
      samples += (DiabetesSample(item[0], item[1]),)

    return samples


# 0 = opel
# 1 = saab
# 2 = bus
# 3 = van
def get_vehicles_dataset(path):
    '''
    Return a list containing samples as sample_class, (tuple of features), id
    Attribute names taken from vehicles.name file'''

    dataframe = p.read_csv(path, names = ['COMPACTNESS', 'CIRCULARITY', 'DISTANCE CIRCULARITY', 'RADIUS RATIO', 'PR.AXIS ASPECT RATIO', 'MAX.LENGTH ASPECT RATIO', 'SCATTER RATIO', 'ELONGATEDNESS','PR.AXIS RECTANGULARITY', 'MAX.LENGTH RECTANGULARITY', 'SCALED VARIANCE 1', 'SCALED VARIANCE 2', 'SCALED RADIUS', 'SKEWNESS 1', 'SKEWNESS 2', 'KURTOSIS 1', 'KURTOSIS 2', 'HOLLOWS RATIO', 'Type'])

    # transpose the dataframe so that it is easier to pick them by just selecting a column
    data = dataframe.T

    # list for storing each individual sample as a list
    list_for_samples = []
    for col in range(data.shape[1]):
        list_for_samples.append(data[col].tolist())

    # total_list = []
    # for k in range(18):
    #     new_list = []
    #     for item in list_for_samples:
    #         new_list.append(item[k])
    #     total_list.append(new_list)
    # # print(total_list)
    #
    # maximums = [max(x) for x in total_list]
    # minimums = [min(x) for x in total_list]
    # for i, item in enumerate(maximums):
    #     print(i+1, minimums[i], item)

    # list for storing samples as sample_class, (tuple of features), id as required by the ml program
    structured_data = []
    for row in list_for_samples:
        structured_data.append([row[-1],tuple(row[0:len(row)-1])])

    # iterate through items in sampels and return the Sample()
    samples = ()
    for item in structured_data:
      samples += (VehiclesSample(item[0], item[1]),)

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

# # Example for housevotes dataset
# samples = get_hv_dataset(housevotes_path)
# for item in samples:
#   print item.class_name, item.attribute
# print len(samples)

# # Example for sonar dataset
# samples = get_sonar_dataset(sonar_path)
# for item in samples:
#    print (item.positive, item.attribute)
# print (len(samples))

# # Example for diabetes dataset
# samples = get_diabetes_dataset(diabetes_path)
# for item in samples:
#    print (item.positive, item.attribute)
# print (len(samples))

# # Example for vehicles dataset
# samples = get_vehicles_dataset(vehicles_path)
# for item in samples:
#    print (item.positive, item.attribute)
# print (len(samples))