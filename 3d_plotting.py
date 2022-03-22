"""
Authors: Kyle Koeller
Date: 03/20/2022

Practice problem 1.21 from Classical Electrodynamics John David Jackson 3rd Edition for the plots that are required.
"""
# import libraries to be used
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def main():
    """
    Calls the main two functions for 2D and the 3D plot making

    :return: 1 plot displaying all graphs corresponding to various fixed y values
    """
    # two_d()
    three_d()


def two_d():
    """
    Produces 2D plot for a wide range of fixed y values
    """
    # initial variables being defined here
    y = [0.1, 0.25, 0.5, 0.6, 0.8]
    x = np.arange(0, 1.001, 0.001)
    m = 355  # this should be infinity but only allowed to go up to 355
    g = 1
    A = (5/4)*g

    # different line styles to be used for plotting
    line_style = [(0, (1, 10)), (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)),
                  (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)),
                  (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
    line_count = 0

    # main loops to loop through the different y values and all the x's and all the m's
    # first loop is for the true psi equation from part b
    for j in y:
        psi_true = []
        for k in range(0, len(x)):
            # make "psi1" be 0 after each successful loop in order to reset the value before appending
            psi1 = 0
            for i in range(0, m):
                cosh = mt.cosh((2 * i + 1))
                psi1 += (mt.sin((2*i + 1) * (mt.pi * x[k]))/((2 * i + 1)**3)) * \
                        (1 - ((cosh * mt.pi * (j - 0.5)) / (cosh * (mt.pi / 2))))
            psi_true.append((16 / (mt.pi ** 2)) * psi1/(4*mt.pi))
        # plots all x values and equivalent psi values given the fixed y value
        plt.plot(x, psi_true, label="True y="+str(j), color="black", linestyle=line_style[line_count])
        line_count += 1

    # this is the estimate psi equation from part a
    for j in y:
        psi_est = []
        for k in range(0, len(x)):
            psi2 = A * (x[k] * (1 - x[k])) * (j * (1 - j))
            psi_est.append(psi2)
        # plots all x values and equivalent psi values given the fixed y value
        plt.plot(x, psi_est, label="Estimate y=" + str(j), color="black", linestyle=line_style[line_count])
        line_count += 1
    plt.xlim(0, 1)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\u03B5_{0}$\u03A6(x,y)")
    plt.grid()
    plt.show()


def three_d():
    """
    Produces an output file of a grid of x and y values with the psi as the height to produce a 3D surface plot outside
    this code

    :return: 2 output files for part a and b from Jackson 1.21 with their respective x-y grid and psi values
    """
    # defining initial variables
    y = np.linspace(0.0, 1.0, 150)
    x = np.linspace(0.0, 1.0, 150)
    m = 355  # this should be infinity but only allowed to go up to 355
    g = 1
    A = (5 / 4) * g

    x_new = []
    y_new = []
    psi_true = []
    psi_est = []
    psi2 = 0
    # first loop gathers all x values for any given y value
    for j in tqdm(range(0, len(y)), "Loop 1"):
        for k in range(0, len(x)):
            # make these values 0 after each loop otherwise they would in theory reach infinity or -infinity
            psi1 = 0
            for i in range(0, m):
                """
                loop through all the m's for part b equation and part a equation
                as I type these comments up "psi2" does not need to be in this specific loop but should be under 
                the k loop
                """
                psi2 = A * (x[k] * (1 - x[k])) * (j * (1 - j))
                # makes the following equation just a little easier to read
                cosh = mt.cosh((2 * i + 1))
                psi1 += (mt.sin((2*i + 1) * (mt.pi * x[k]))/((2 * i + 1)**3)) * \
                        (1 - ((cosh * mt.pi * (y[j] - 0.5)) / (cosh * (mt.pi / 2))))
            # append all values to their respective lists
            x_new.append(x[k])
            y_new.append(y[j])
            psi_true.append((16/mt.pi**2)*(psi1/(4*mt.pi)))
            psi_est.append(psi2)

    # second loop gathers all y values for any given x value
    # same comments as the previous grouping of loops
    for k in tqdm(range(0, len(x)), "Loop 2"):
        for j in range(0, len(y)):
            psi1 = 0
            for i in range(0, m):
                psi2 = A * (x[k] * (1 - x[k])) * (j * (1 - j))
                cosh = mt.cosh((2 * i + 1))
                psi1 += (mt.sin((2*i + 1) * (mt.pi * x[k]))/((2*i + 1)**3)) * \
                        (1 - ((cosh * mt.pi * (y[j] - 0.5)) / (cosh * (mt.pi / 2))))
            x_new.append(x[k])
            y_new.append(y[j])
            psi_true.append((16/mt.pi**2)*(psi1/(4*mt.pi)))
            psi_est.append(psi2)

    # output the lists into dataframes for easy output transfer
    df_true = pd.DataFrame({
        "x": x_new,
        "y": y_new,
        "psi_true": psi_true
    })

    df_est = pd.DataFrame({
        "x": x_new,
        "y": y_new,
        "psi_estimate": psi_est
    })

    # save the dataframes to output text files for further 3d plotting
    df_true.to_csv("psi_true.txt", index=False)
    df_est.to_csv("psi_est.txt", index=False)
    print("Finished Saving")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    color_map = plt.get_cmap('spring')
    ax.scatter3D(x_new, y_new, psi_est, cmap=color_map, c=psi_est)
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_zlabel('$\u03A6_{estimate}$', fontweight='bold')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter3D(x_new, y_new, psi_true, cmap=color_map, c=psi_true)
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_zlabel('$\u03A6_{true}$', fontweight='bold')
    plt.show()


if __name__ == '__main__':
    main()
