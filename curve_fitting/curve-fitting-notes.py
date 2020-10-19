from sympy import *
from sympy.plotting import plot
init_printing(use_latex='mathjax')

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# Sympy
x = symbols('x')

x_points = [0,1,2,3,4,5]
y_points = [0.5, 2, 3, 2.8, 2, 0.4]

model = x**2

x = np.linspace(0, 10, 30)
y = np.sin(x)


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

fig = ax.scatter(x_points, y_points, color='black');
# plt.close()


x_list = np.linspace(0, 10, 100)

lam_x = lambdify(x, model)

"""
x = symbols('x')

f = x**2
display(f)
f_mark = Derivative(f)
display(f_mark)
f_diff = f.diff(x)
display(f_diff)

p1 = plot(f, show=False)

p1.show()
"""