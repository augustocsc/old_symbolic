from SymbolicMathematics.src.utils import AttrDict
from SymbolicMathematics.src.envs import build_env
import re
import math

params = params = AttrDict({

    # environment parameters
    'env_name': 'char_sp',
    'int_base': 10,
    'balanced': False,
    'positive': True,
    'precision': 10,
    'n_variables': 1,
    'n_coefficients': 0,
    'leaf_probs': '0.75,0,0.25,0',
    'max_len': 512,
    'max_int': 5,
    'max_ops': 15,
    'max_ops_G': 15,
    'clean_prefix_expr': True,
    'rewrite_functions': '',
    'tasks': 'prim_fwd',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1'
})

env = build_env(params)

def prefix_to_infix(s, opt_dict = None):
    try:
    
        exp = env.prefix_to_infix(s)
        exp_symp = env.infix_to_sympy(exp)

    #exp_symp = exp_symp.evalf()
        return exp_symp
    except:
        return None
def map_expression(x):

    # Define a regular expression pattern to match 'x' not surrounded by word characters
    pattern = r'\bx\b'

    # Use the re.sub function to perform the replacement
    result = re.sub(pattern, 'x[0]', x)
    
    return result

def compute(expression, x):
    x = x[0]
    expression = map_expression(expression)
    if expression is not None:
        return expression.evalf(subs={'x[0]':x})
    else:
        return None
import re

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False



