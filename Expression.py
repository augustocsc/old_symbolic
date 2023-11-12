from utils import prefix_to_infix
import math
from sympy import lambdify
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

opt_dict={'sqrt':'np.sqrt','sin':'np.sin','cos':'np.cos','tan':'np.tanh'}

class Expression:
  def __init__(self, prefix, data, c=None, method='nrmse_dso',
             optr_dict=opt_dict):

    # Get the prefix

    self.prefix = [x for x in prefix if x.strip(' ')]

    #print("self.prefix={}".format(self.prefix))

    self.infix = prefix_to_infix(self.prefix, opt_dict)
    
    if self.infix is None:
        self.pred_y = None
        self.score = 0
    else:
        self.y_pred = self.eval_expr(data.x)
        self.y_pred = np.array(self.y_pred).flatten()
        
        try:
          self.r2 = r2_score(data.y, self.y_pred)
        except:
          self.r2 = 0
        
        self.score = self.compute_reward(data.y, method=method)

        if math.isnan(self.score):
            self.score = 0

    #print("{} {:.4f}".format(self.infix, self.score))
    #print("self.score={}\n".format(self.score))


  def __str__(self):
    return f"{self.infix}"

  def compute_reward(self, y, method='nrmse_dso'):
    try:
      if method == 'nrmse_dso':
        y = np.array(y).flatten()
        sigma = np.std(y)
        reward = self.nrmse(y, self.y_pred)
        return 1 / (1 + reward/sigma)
      if method == 'nrmse':
        return self.nrmse(y, self.y_pred)
      if method == 'r2':
        return self.r2(y, self.y_pred)
    except Exception as error:
      print(f'Error: {error}')
      return 0

  def r2(self, y, y_pred):
    try:
      reward = r2_score(y, y_pred)
    except:
      reward = 0
    return reward
  
  def nrmse(self, y, y_pred):
    reward = mean_squared_error(y, y_pred, squared=False)
    return reward
  
  def eval_expr(self, x):
    try:
      f = lambdify('x', self.infix, "numpy")
    except Exception as error:
      print(f'Expression {self.infix} cannot be evaluated. Error: {error}')
      return math.nan
    
    return f(x)