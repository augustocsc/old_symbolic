import math

def prefix_to_infix(expression):
    stack = []

    def is_operator(token):
        return token in {'sum', 'add', 'mul', 'div', 'abs', 'inv', 'sqrt', 'log', 'exp', 'cos', 'sin', 'tan', 'pow2', 'pow3'}

    def is_operand(token):
        return not is_operator(token)

    def get_python_operator(operator, is_unary=False):
        operator_mapping = {
            'sum': 'sum',
            'add': '+',
            'mul': '*',
            'div': '/',
            'abs': 'abs',
            'inv': '1/',
            'sqrt': 'math.sqrt',
            'log': 'math.log',
            'exp': 'math.exp',
            'cos': 'math.cos',
            'sin': 'math.sin',
            'tan': 'math.tan',
            'pow': '**',
            'pow2': 'pow',
            'pow3': 'lambda x: pow(x, 3)'
        }
        if is_unary:
            return operator_mapping.get(operator, operator)
        else:
            return operator_mapping.get(operator, operator)

    def get_infix_representation(operator, operands):
        python_operator = get_python_operator(operator)
        if operator in {'abs', 'inv', 'sqrt', 'cos', 'sin', 'tan', 'pow2'}:
            return f'{python_operator}({operands[0]})'
        elif operator == 'log':
            return f'{python_operator}({operands[0]}, {operands[1]})'
        elif operator == 'pow3':
            return f'{python_operator}({operands[0]}, 3)'
        else:
            return f'({operands[0]} {python_operator} {operands[1]})'

    for token in reversed(expression):
        if is_operand(token):
            stack.append(token)
        elif is_operator(token):
            operator = token
            is_unary = len(stack) == 1  # Check if the operator is unary
            operand1 = stack.pop()
            if is_unary:
                infix_representation = f'{get_python_operator(operator, True)}({operand1})'
            else:
                operand2 = stack.pop()
                infix_representation = get_infix_representation(operator, [operand2, operand1])
            stack.append(infix_representation)

    return stack[0]

# Example
expression = ['add', 'mul', 'c', 'mul', 'pow', 'x4', 'c', 'add', 'x4', 'cos', 'add', 'mul', 'c', 'sqrt', 'x4', 'add', 'mul', 'c', 'x4', 'mul', 'c', 'pow', 'x4', 'c']
infix_expression = prefix_to_infix(expression)
print(infix_expression)
#print(eval(infix_expression))

