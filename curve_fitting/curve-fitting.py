
# Clear defined variables
from IPython.display import display, Latex
from sympy import *
from sympy.plotting import plot
init_printing(use_latex='mathjax')

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# Symbols used
j, k, n, v, x, y = symbols('j k n v x y')

# Data points
x_points = [0,1, 2, 3, 4,5,6,7,8,9]
y_points = [0.5, 4.5, 1, -2, -4, -2, -1.5, -0.3, 1.2, 2.0]

# Model to use
num_free_vars = 2
model = Indexed(v, 1)*Indexed(x, k)**1 + Indexed(v, 2)*Indexed(x, k)**0
display(model)


E = Sum((model-Indexed(y,k))**2,(k,0,len(x_points) -1))
display(E)

for var_num in range(num_free_vars):
    eq = E.diff(Indexed(v, var_num))
    display(eq)

# eqA = E.diff(a)
# display(eqA)
# f_A = lambdify((x, y, a, b, c, d, e), eqA)
# list_y_A = f_A(x_points, y_points, a, b, c, d, e)
# display(list_y_A)

# eqB = E.diff(b)
# display(eqB)
# f_B = lambdify((x, y, a, b, c, d, e), eqB)
# list_y_B = f_B(x_points, y_points, a, b, c, d, e)
# display(list_y_B)

# eqC = E.diff(c)
# display(eqC)
# f_C = lambdify((x, y, a, b, c, d, e), eqC)
# list_y_C = f_C(x_points, y_points, a, b, c, d, e)
# display(list_y_C)

# eqD = E.diff(d)
# display(eqD)
# f_D = lambdify((x, y, a, b, c, d, e), eqD)
# list_y_D = f_D(x_points, y_points, a, b, c, d, e)
# display(list_y_D)

# eqE = E.diff(e)
# display(eqE)
# f_E = lambdify((x, y, a, b, c, d, e), eqE)
# list_y_E = f_E(x_points, y_points, a, b, c, d, e)
# display(list_y_E)

# # Solving equations
# solved = solve((list_y_A, list_y_B, list_y_C, list_y_D, list_y_E), (a, b, c, d, e))
# print(solved)

# # Assigning values to symbols
# a = solved[a]
# b = solved[b]
# c = solved[c]
# d = solved[d]
# e = solved[e]

# # Generating points for model plot
# x_list = np.linspace(-10, 100, 10000)
# unindexed_model = a*x**4 + b*x**3 + c*x**2 + d*x**1 + e*x**0
# y_generator = lambdify((x), unindexed_model)

# display(unindexed_model)
# y_list = y_generator(x_list)

# # Setup figure
# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# fig = ax.set_xlim((-1,6))
# fig = ax.set_ylim((-5,10))
# # Display figure
# fig = ax.scatter(x_points, y_points, color='black');
# fig = ax.plot(x_list, y_list)
# # plt.close() # Don't display
# plt.show()