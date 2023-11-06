import argparse
import numpy as np
import avaliation
from datetime import datetime
import os
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--expr_id", type=str, default= datetime.now().strftime('%Y%m%d_%H%M'),
                        help="Experiment identification")

    return parser

def initialize_exp(params):
    
    #check if ./experiment/ exists
    if not os.path.exists('./experiment/'):
        os.mkdir('./experiment/')
    
    #check if ./experiment/expr_id/ exists
    if not os.path.exists('./experiment/'+params.expr_id):
        os.mkdir('./experiment/'+params.expr_id)

def main(params):
    avaliation.main(params)

if __name__ == '__main__':
    
    parser = get_parser()
    params = parser.parse_args()
    main(params)