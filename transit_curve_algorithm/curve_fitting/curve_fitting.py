"""
Fit a given model to a dataset of (x, y) points
"""

import re
import random
import numpy as np
import sympy as sym
from sympy.plotting import plot
import math as Math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Symbols used
j, k, n, v, x, y, a = sym.symbols('j k n v x y a')


def calc_R2(model, x_points, y_points):  # Calculate R^2
    model = model.subs(x, sym.Indexed(x, k))    
    R2_model = sym.sqrt(1/len(x_points)*sym.Sum(((model) -
                                                 sym.Indexed(y, k))**2, (k, 0, len(x_points) - 1)))
    f_R2 = sym.lambdify((x, y), R2_model)
    return f_R2(x_points, y_points)


def fit_model(str_model, x_points, y_points):  # Fit a model to a dataset of (x, y) points
    # Interpret and evaluate the given model
    still_searching = True
    model = str_model.replace('x', 'sym.Indexed(x, k)')
    free_vars = []
    free_vars_indices = []
    while still_searching:
        match = re.search(r"v_\d+", model)
        if not match:
            still_searching = False
            continue
        matching_text = match.group()
        var_num = re.search(r"\d+", matching_text).group()
        model = model[:match.span()[0]] + "sym.Indexed(v," + \
            str(var_num) + ")" + model[match.span()[1]:]
        free_vars.append(eval("sym.Indexed(v," + str(var_num) + ")"))
        free_vars_indices.append(int(var_num))

    model = eval(model)
    # Symbols used but this time declared inside function
    j, k, n, v, x, y, a = sym.symbols('j k n v x y, a')

    # Number of free variables in the model
    num_free_vars = len(free_vars)

    # To be minimized
    E = sym.Sum((model-sym.Indexed(y, k))**2, (k, 0, len(x_points) - 1))

    solved_eqs = []  # List of partially solved equations
    # to_display("Starting now", "debug")
    for var_num in free_vars_indices:
        print("v_" + str(var_num))  # Print this iteration number
        eq = E.diff(sym.Indexed(v, var_num))  # Find the derivative
        # Lambdify the newly created equation
        f = sym.lambdify((x, y, *free_vars), eq)
        # Solve the equation as much as possible
        solved_eq = f(x_points, y_points, *free_vars)
        # Add the soved equation to the list of solved equations
        solved_eqs.append(solved_eq)

    # Solve all the equations for all the free variables
    solved = sym.solve((solved_eqs), (free_vars),
                       simplify=False)    

    # Create list of evenly distibuted x-points
    x_list = np.linspace(np.min(x_points) - 1, np.max(x_points) + 1, 10000)

    # Find variables in model str and replace with solved[sym.Indexed(v,n)]
    still_searching = True
    solved_model = str_model
    while still_searching:
        match = re.search(r"v_\d+", solved_model)
        if not match:
            still_searching = False
            continue
        matching_text = match.group()
        var_num = pattern = r"\d+"
        var_num = re.search(r"\d+", matching_text).group()
        if not sym.Indexed(v, var_num) in solved:
            print("excluding {}".format(var_num))
            solved_model = solved_model[:match.span(
            )[0]] + "0" + solved_model[match.span()[1]:]
            continue
        solved_model = solved_model[:match.span(
        )[0]] + "solved[sym.Indexed(v," + str(var_num) + ")]" + solved_model[match.span()[1]:]

    # Evaluating the solved model
    solved_model_eval = eval(solved_model)

    # Generate y-coords for each x-coord
    y_generator = sym.lambdify((x), solved_model_eval)
    y_list = y_generator(x_list)

    # Setup figure
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    fig = ax.set_xlim((np.min(x_points) - 0.02, np.max(x_points) + 0.02))
    fig = ax.set_ylim((np.min(y_points) - 0.02, np.max(y_points) + 0.02))
    
    # R^2
    print("R^2:")
    print(calc_R2(solved_model_eval, x_points, y_points), "R2")

    # Return model
    return(x_list, y_list)



# Generate a taylor polynomium
def n_degree_poly(n):
    model = ""
    for i in range(1, n+2):
        model += "+v_" + str(i) + "*x**" + str(n+1-i)
    return model
