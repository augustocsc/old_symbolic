#python main.py --save_dir='500_epochs' --epochs=300
import argparse
import numpy as np
from datetime import datetime
from Agent import Agent
import os
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--save_dir", type=str, default= datetime.now().strftime('%Y%m%d_%H%M'),
                        help="Experiment identification")
    parser.add_argument("--experiment_file", type=str, default="data/expressions/test_expr.csv",
                        help="File containing the expressions to be evaluated")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stop_condition", type=str, default="epoch", choices=["epoch", "reward", "mean", "all"], 
                        help="Stop condition") #TODO implement a stop condition based on the mean of the rewards and a way to set the values of the stop condition
    
    # training parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16,
                        help="Mini batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1.5e-5,
                        help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.99,
                        help="Learning rate decay")
    
    return parser

def initialize_exp(params):
    
    #check if ./experiment/ exists    
    if not os.path.exists('./experiment/'):
        os.mkdir('./experiment/')
    
    #check if ./experiment/save_dir/ exists
    if not os.path.exists(f'./experiment/{params.save_dir}'):
        os.mkdir(f'./experiment/{params.save_dir}')
        

def main(params):
    agent = Agent(params)
    agent.train()

if __name__ == '__main__':
    print('Starting main.py')
    parser = get_parser()
    params = parser.parse_args()
    initialize_exp(params)
    params.save_dir = f'./experiment/{params.save_dir}'
    main(params)