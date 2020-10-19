import re
import random
import numpy as np
import sympy as sym
from sympy.plotting import plot
import math as Math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from curve_fitting.curve_fitting import fit_model, n_degree_poly

# Data points loaded from csv
x_points = []
y_points = []
print("Loading CSV ...")
arr = np.genfromtxt('empiri/folded_data.csv', delimiter=',')
for i in range(len(arr)):
    point = arr[i]
    if point[0] > 3.2 and point[0] < 3.55:
        if i > 0:
            variance = 0.006
            if point[1] < arr[i-1][1] + variance and point[1] > arr[i-1][1] - variance:
                x_points.append(point[0])
                y_points.append(point[1])
        else:
            x_points.append(point[0])
            y_points.append(point[1])
print("CSV loaded")

# Show the raw data points
# plt.scatter(x_points, y_points)
# plt.show()
n = 14

for i in range(4, 5):
    model = fit_model(n_degree_poly(i), x_points, y_points)
    plt.scatter(x_points, y_points, color='black')
    plt.plot(model[0], model[1])
    plt.show()