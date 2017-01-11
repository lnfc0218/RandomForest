import pandas as p
import numpy

glass_path = "Datasets/glass.data"
bc_path = "Datasets/breast-cancer-wisconsin.data"

p.set_option('display.max_row', 1000)


class GlassSample:
    "Represenation of data samles"

    def __init__(self, sample_class, values, identity):
        self.sample_class = sample_class
        self.attribute = dict(zip(glass_attributes, values))
        self.identity = identity

class BCSample:
    "Represenation of data samles"

    def __init__(self, sample_class, values, identity):
        self.sample_class = sample_class
        self.attribute = dict(zip(bc_attributes, values))
        self.identity = identity


class Attribute:
    "Label for each attribute"

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __repr__(self):
        return self.name


glass_attributes = (Attribute('A1', tuple(numpy.linspace(1.5112,1.5339).tolist())),
              Attribute('A2', tuple(numpy.linspace(10.73,17.38).tolist())),
              Attribute('A3', tuple(numpy.linspace(0 , 4.49).tolist())),
              Attribute('A4', tuple(numpy.linspace(0.29, 3.5).tolist())),
              Attribute('A5', tuple(numpy.linspace(69.81, 75.41).tolist())),
              Attribute('A6', tuple(numpy.linspace(0, 6.21).tolist())),
              Attribute('A7', tuple(numpy.linspace(5.43, 16.19).tolist())),
              Attribute('A8', tuple(numpy.linspace(0, 3.15).tolist())),
              Attribute('A9', tuple(numpy.linspace(0, 0.51).tolist())))


bc_attributes = (Attribute('A1', tuple(numpy.linspace(3,4).tolist())),
              Attribute('A2', tuple(numpy.linspace(3,2).tolist())),
              Attribute('A3', tuple(numpy.linspace(2,3).tolist())),
              Attribute('A4', tuple(numpy.linspace(2,1).tolist())),
              Attribute('A5', tuple(numpy.linspace(3,2).tolist())),
              Attribute('A6', tuple(numpy.linspace(0, 3).tolist())),
              Attribute('A7', tuple(numpy.linspace(5, 16).tolist())),
              Attribute('A8', tuple(numpy.linspace(0, 3).tolist())))


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
      samples += (GlassSample(item[0], item[1], item[2]),)

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
    for item in structured_data:
      samples += (GlassSample(item[0], item[1], item[2]),)

    return samples
 

# Example for glass dataset
samples = get_glass_dataset(glass_path)
print len(samples)
for item in samples:
   print item.sample_class, item.attribute, item.identity


# Example for bc dataset
samples = get_bc_dataset(bc_path)
for item in samples:
   print item.sample_class, item.attribute, item.identity
print len(samples)
