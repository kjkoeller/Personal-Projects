"""
Author: Kyle Koeller
Date Created: 03/04/2022

This program fits a set of data with a line, quadratic, and polynomial fit of degree 3
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


def line_fit():
    """
    Create a linear fit by hand and than use scipy to create a polynomial fit given an equation along with their
    respective residual plots.

    :return: Fits and residuals for a given data set
    """

    # read in the text file
    df = pd.read_csv("total_minimums.txt", header=None, delim_whitespace=True)

    x = df[1]
    y = df[2]
    y_err=df[3]
    N = len(x)

    # scipy_func(x, y)

    # start finding all the sums that are required in a linear fit
    sum_xy_list = []
    count = 0
    for i in x:
        sum_xy_list.append(i*y[count])
        count += 1

    sum_xy = sum(sum_xy_list)
    sum_x = sum(x)
    sum_y = sum(y)

    sum2_x = sum_x**2
    sum2_y = sum_y**2

    x2 = []
    y2 = []
    count = 0
    for i in x:
        y2.append(y[count]**2)
        x2.append(i**2)
        count += 1
    sum_x2 = sum(x2)
    sum_y2 = sum(y2)

    # finally make the equation for your a and b values in the y = ax + b equation
    a = (N*sum_xy - sum_x*sum_y)/(N*sum_x2-sum2_x)  # slope
    b = (sum_y*sum_x2 - sum_x*sum_xy)/(N*sum_x2 - sum2_x)  # intercept
    # r squared value absolute valued for between 0 <= r <= 1
    r_line = np.abs((N*sum_xy - sum_x*sum_y)/(np.sqrt(N*sum_x2-sum2_x)*np.sqrt(N*sum_y2-sum2_y)))
    print("Linear R^2: " + str(r_line))
    line = a*x + b

    # find the residuals for the linear fit line
    residuals_line = []
    count = 0
    for i in x:
        y_x = a*i + b
        residuals_line.append(y_x - y[count])
        count += 1

    s = UnivariateSpline(x, y)
    xs = np.linspace(x.min(), x.max(), 1000)
    ys = s(xs)

    # numpy curve fit
    degree = 2
    model = np.poly1d(np.polyfit(x, y, degree))
    print("R Squared value of Quadratic Fit: "+str(adjR(x, y, degree)))
    print("Coefficients of 3rd Order Polynomial: "+str(s.get_coeffs()))

    # plot the main graph with both fits (linear and poly) onto the same graph
    plt.errorbar(x, y, yerr=y_err, fmt="o", color="black")
    plt.plot(x, line, color="black", label="Linear it: a="+str(a)+", b="+str(b))
    plt.plot(xs, model(xs), linestyle="dotted", color="black", label="polynomial fit of degree " + str(degree))
    plt.plot(xs, ys, linestyle="dashed", color="black", label="polynomial fit of degree "+str(len(s.get_coeffs())-1))

    # make the legend always be in the upper right hand corner of the graph
    plt.legend(loc="upper right")
    plt.xlabel("Epoch Number")
    plt.ylabel("O-C (days)")
    plt.title("NSVS 254037 O-C")
    plt.grid()
    plt.show()


def adjR(x, y, degree):
    """
    Finds the R Squared value

    :param x: x data points
    :param y: y data points
    :param degree: polynomial degree
    :return:
    """
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results["r_squared"] = 1 - (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))

    return results


if __name__ == '__main__':
    line_fit()
