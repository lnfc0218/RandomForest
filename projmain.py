import pandas as p
import numpy

path = "Datasets/glass.data"
p.set_option('display.max_row', 1000)


class GlassSample:
    "Represenation of data samles"

    def __init__(self, identity, values, type):
        self.identity = identity
        self.attribute = dict(zip(glass_attributes, values))
        self.type = type


class Attribute:
    "Label for each attribute"

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __repr__(self):
        return self.name


glass_attributes = (Attribute('RI', tuple(numpy.linspace(1.5112,1.5339).tolist())),
              Attribute('Na', tuple(numpy.linspace(10.73,17.38).tolist())),
              Attribute('Mg', tuple(numpy.linspace(0 , 4.49).tolist())),
              Attribute('Al', tuple(numpy.linspace(0.29, 3.5).tolist())),
              Attribute('Si', tuple(numpy.linspace(69.81, 75.41).tolist())),
              Attribute('K', tuple(numpy.linspace(0, 6.21).tolist())),
              Attribute('Ca', tuple(numpy.linspace(5.43, 16.19).tolist())),
              Attribute('Ba', tuple(numpy.linspace(0, 3.15).tolist())),
              Attribute('Fe', tuple(numpy.linspace(0, 0.51).tolist())))


def read_glass_dataset(path):
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
    print structured_data
    samples = ()
    for item in structured_data:
      samples += (GlassSample(item[2], item[1], item[0]),)

    return samples
 

# Example for glass dataset
samples = read_glass_dataset(path)
for item in samples:
  print item.identity, item.attribute, item.type
