import numpy as np
import pandas as pd
import re
import time

#TODO: every experiment should print a set of informations that can be saved in a file
#cases of use
#case 1: data is available
#case 2: only expression is available
#case 3: expression and daatset are available
#case 4: expression and interval are available
class Experiment:
  def __init__(self, x=None, y=None, experiment=None, save=True, save_folder='data/input/'):
    experiment = experiment.replace(np.nan, None)

    # Set the seed for reproducibility
    self.x = x
    self.y = y
    if self.y is not None:
      self.sigma = np.std(self.y)
     
    self.save = save
    
    if experiment is not None:
      self.name = experiment['name'] if experiment['name'] is not None else "experiment_{}".format(time.time())
      self.expression = self.map_expression(experiment['expression']) if experiment['expression'] is not None else "x**2"
      self.dataset = str(experiment['dataset']) if experiment['dataset'] is not None else "{}.csv".format(self.name.replace(' ', ''))
      self.dataset = save_folder + self.dataset
      self.low = -1 if experiment['low'] == None else experiment['low']
      self.high = 1 if experiment['high'] == None else experiment['high']
      self.size = 100 if experiment['size'] == None else eval(experiment['size'])   #TODO ensure that size works both with tuples and integers
      self.noise = 0 if experiment['noise'] == None else experiment['noise']
      self.seed = 42 if experiment['seed'] == None else int(experiment['seed'])

      self.load_experiment_data()

    elif x is None:
      self.x =  self.create_dataset()
      if save:
        self.save_dataset()
    else:
        self.x = x
        #if no y is provide return an error
        if y is None:
          raise ValueError('y cannot be None')
        else:
          self.y = y
          self.sigma = np.std(y)
    
  def create_dataset(self):
    print("Creating dataset for {}".format(self.name))
    # Set the seed for reproducibility
    np.random.seed(self.seed)
    self.x =  np.random.uniform(low=self.low, high=self.high, size=self.size)
    #try to evaluate the expression, if it fails, return an error
    try:
      self.y = eval(self.expression.replace('x[', 'self.x['))
      self.y = self.y.flatten()
      self.sigma = np.std(self.y)
    except Exception as error:
      raise ValueError(f'Expression {self.expression} cannot be evaluated. Error: {error}')
    
    if self.save:
      self.save_dataset()
    return self.x

  def save_dataset(self):
    #save the dataset in the filename
    df = pd.DataFrame({'x': self.x.flatten(), 'y': self.y.flatten()})
    df.to_csv(self.dataset, index=False)
    print("Dataset saved in {}".format(self.dataset))

  def __str__(self):
      print(self.__dict__)

  def map_expression(self, x):

      # Define a regular expression pattern to match 'x' not surrounded by word characters
      pattern = r'\bx\b'

      # Use the re.sub function to perform the replacement
      result = re.sub(pattern, 'self.x', x)
      
      return result
  def load_experiment_data(self):
    
    print("Loading experiment data {}".format(self.name))

    print("Loading dataset from {}".format(self.dataset))
    #try to read the dataset from the experiment, if it fails, create a new dataset
    try:
      data = pd.read_csv(self.dataset)
      self.experiment.x = data['x']
      self.experiment.y = data['y']
    except:
      print("Dataset {} not found".format(self.dataset))
      self.create_dataset()

    return self