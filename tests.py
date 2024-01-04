
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, exp, cos, sin, tan, log, pi, e, sinh, cosh, tanh, arccos, arcsin, arctan, arctan2, arcsinh, arccosh, arctanh
import math
from sympy import lambdify
import numpy as np
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

def eval_expr(expression, x):
    try:
        #replace acos with arccos
        expression = expression.replace('acos', 'arccos')
        #replace asin with arcsin
        expression = expression.replace('asin', 'arcsin')
        #replace atan with arctan
        expression = expression.replace('atan', 'arctan')
        f = lambdify('x', expression, "numpy")
        return f(x)
    except Exception as error:
        print(f'Expression {expression} cannot be evaluated. Error: {error}')
        return math.nan

    
expression_folder = "./data/expressions/"
input_folder = "./data/input/"
expression_file = "all_expr.csv"
result_folder = "./experiment/1000_points/"

#creating a report folder
report_folder = os.path.join(result_folder, "report")
if not os.path.exists(report_folder):
    os.mkdir(report_folder)
df = pd.read_csv(os.path.join(expression_folder, expression_file), index_col=False)

for _, expression in df.iterrows():
    
    #load input data
    input_file = os.path.join(input_folder, expression['dataset'])
    input_df = pd.read_csv(input_file, dtype={'x': float, 'y': float})
    
    #convert the column x and y to float
    input_df['x'] = input_df['x'].apply(convert_to_float)
    input_df['y'] = input_df['y'].apply(convert_to_float)
    
    print(f'Processing {expression["name"]}')
    #load log file
    log_file = os.path.join(result_folder, expression['name'], "log_saved.csv")
    log_df = pd.read_csv(log_file, names=['id', 'expression', 'reward', 'r2'])

    #prepare data for plotting
    log_df, top_5 = prepare_data(log_df)

    
    #create a numpy array from the range expression['low'] and expression['high'] with expression['size'] number of elements
    _,size = eval(expression['size'])
    x = np.linspace(expression['low'], expression['high'], size)
    
    #converting top_5 dataframe into a dictionary
    top_5_dict = top_5.to_dict('records')

    #iterating through top 5 dict evaluating the expressions and storing the result in a dictionary
    for i, expr in enumerate(top_5_dict):
        print(expr['expression'])
        top_5_dict[i]['y'] = eval_expr(expr['expression'], x)


    #group log_df 128 rows at a time and take mean and max of reward and r2
    log_df_short = log_df.groupby(np.arange(len(log_df))//128).agg({
                                                              'reward': ['mean', 'max'], 
                                                              'r2': ['mean', 'max']})
    #for each 128 rows, count the number of invalid expressions (reward = 0) and store in a list
    invalid_expr = []
    for i in range(0, len(log_df), 128):
        try:
            invalid_expr.append(log_df[i:i+128]['reward'].value_counts()[0])
        except:
            invalid_expr.append(0)
    #add the list as a column to log_df_short
    log_df_short['invalid_expr'] = invalid_expr
    
    log_df_short.reset_index(drop=True, inplace=True)

    # Create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    #the title of the figure is the name of the expression smaller and the expression bigger bellow
    fig.suptitle(expression['name'], fontsize=16)
    fig.text(0.5, 0.04, expression['expression'], ha='center', fontsize=14)
    
    

    # Plot reward max and mean in one chart and r2 max in another chart
    axs[0].plot(log_df_short.index, log_df_short['reward']['mean'], 'b--', label='reward mean', color='royalblue')
    axs[0].plot(log_df_short.index, log_df_short['reward']['max'], 'b-', label='reward max', color='darkblue')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Epochs')
    axs[0].legend(loc="upper right")

    axs[1].plot(log_df_short.index, log_df_short['r2']['max'], 'r-', label='r2 max', color='limegreen')
    axs[1].set_ylabel('R2')
    axs[1].set_xlabel('Epochs')
    axs[1].legend(loc="upper right")

    # Plot invalid expressions
    axs[2].plot(log_df_short.index, log_df_short['invalid_expr'], 'b-', label='invalid expressions', color='tomato')
    axs[2].set_ylabel('Invalid Expressions')
    axs[2].set_xlabel('Epochs')
    axs[2].legend(loc="upper right")

    # Plot top 5 expressions
    axs[3].scatter(input_df['x'], input_df['y'], label='input data', color="darkorange", )
    for i, expr in enumerate(top_5_dict):
        axs[3].plot(x, expr['y'], label=expr['expression'])
    axs[3].legend(loc="upper right")
    axs[3].set_ylabel('Top 5 Expressions')
    axs[3].set_xlabel('Epochs')

    # Save the combined figure
    combined_file_path = os.path.join(report_folder, expression['name'] + "_combined_plots.png")
    fig.savefig(combined_file_path)

    # Show the combined figure
    plt.show()

    # Print the path of the saved file
    print(f"Combined plots saved to: {combined_file_path}")


'''
Create two charts one on top and other bellow. 
The first one displays reward (blue) and r2 (orange) max and mean values. 
The mean values should be a dotted line and the max a solid one. 
This chart also has a bar chart with the number of invalid expressions. 
The left y-axis goes from 0 to 1, representing the reward and the right side from 0 to 128 
representing the number of invalid expressions. Finally, the x-axis is the epoch.

The second chart (bellow) plot the top 5 expressions and the goal expression (expression['expression']). The best one should be in green, 
the others more grey with 70% of opacity and the goal blue. The goal points (input_df) should points in orange.


    # Plot reward max and mean in one chart and r2 max in another chart
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Reward vs Epochs')
    axs[0].plot(log_df_short.index, log_df_short['reward']['mean'], 'b--', label='reward mean')
    axs[0].plot(log_df_short.index, log_df_short['reward']['max'], 'b-', label='reward max')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Epochs')
    axs[0].legend(loc="upper right")
    axs[1].plot(log_df_short.index, log_df_short['r2']['max'], 'r-', label='r2 max')
    axs[1].set_ylabel('R2')
    axs[1].set_xlabel('Epochs')
    axs[1].legend(loc="upper right")
    fig.savefig(os.path.join(report_folder, expression['name'] + "_reward_r2_vs_epochs.png")) # Save the figure

        #plot the invalid expressions
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(log_df_short.index, log_df_short['invalid_expr'], 'b-', label='invalid expressions')
    ax.set_ylabel('Invalid Expressions')
    ax.set_xlabel('Epochs')
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(report_folder, expression['name'] + "_invalid_expr.png")) # Save the figure

        #plot top 5 expressions
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(input_df['x'], input_df['y'], label='input data', color="darkorange", )
    for i, expr in enumerate(top_5_dict):
        ax.plot(x, expr['y'], label=expr['expression'])
    ax.legend(loc="upper right")
    # Save the figure
    fig.savefig(os.path.join(report_folder, expression['name'] + "_top_5.png"))
'''
