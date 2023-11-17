
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
this report returns the following:
    A figure with 4 subplots:
        1. mean reward and r2 vs epochs
        2. max reward and r2 vs epochs
        3. the input data and the predicted line for the top 5 distinct expressions with highest reward
        4. 
'''
#method to convert a pandas series from string to float, if not possible replace with 0
def convert_to_float(x):
    try:
        return float(x)
    except:
        return 0
#method to get the row of top 5 distinct expressions with highest reward
def get_top_5(log_df):
    #sort log_df by reward in descending order
    log_df = log_df.sort_values(by=['reward'], ascending=False)
    #get top 5 distinct expressions
    top_5 = log_df.drop_duplicates(subset=['expression']).head(5)
    return top_5

#method to prepare the data for plotting
def prepare_data(log_df):
    #replace nan in expression and r2 with -1
    log_df['expression'].fillna(-1, inplace=True)
    log_df['r2'].fillna(-1, inplace=True)

    #replacing tensor(x) to only x in reward and r2 columns
    log_df['reward'] = log_df['reward'].apply(lambda x: x[7:-1])

    #converting reward and r2 to float
    log_df['reward'] = log_df['reward'].astype(float)
    
    #using convert_to_float method to convert r2 to float
    log_df['r2'] = log_df['r2'].apply(convert_to_float)

    #get top 5 distinct expressions with highest reward
    top_5 = get_top_5(log_df)

    return log_df, top_5
expression_folder = "./data/expressions/"
expression_file = "all_expr.csv"
result_folder = "./experiment/500_epochs/"
df = pd.read_csv(os.path.join(expression_folder, expression_file))
for _, expression in df.iterrows():
    log_file = os.path.join(result_folder, expression['name'], "log_saved.csv")
    log_df = pd.read_csv(log_file, names=['id', 'expression', 'reward', 'r2'])

    #prepare data for plotting
    log_df, top_5 = prepare_data(log_df)

    #group log_df 128 rows at a time and take mean and max of reward and r2
    log_df = log_df.groupby(np.arange(len(log_df))//128).agg({'reward': ['mean', 'max'], 'r2': ['mean', 'max']})


    break
