import pandas as pd
from Experiment import Experiment

#read the file data.csv
data = pd.read_csv('data/test_expr.csv')
results = []

#for each line in data pandas dataframe
for _, line in data.iterrows():    
    experiment = Experiment(experiment=line)