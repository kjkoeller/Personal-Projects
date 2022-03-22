"""
Author: Kyle Koeller
Date: 10/27/2020

Program to test the Lane-Emden equation via analytical solutions and a Runge-Kutta 4th order integration scheme
"""

# import libraries
import matplotlib.pyplot as plt
import runge_kutta
import statistics as st
import math as mt


def main():
    # define integration variables
    n = [0, 1, 3/2, 3, 5]  # value of the polytropic index
    t_step = 0.05  # Change the time step of integration, decent value is 0.05
    for i in n:
        xis, thetas = rk_integrate(i, t_step)
        # plot the known analytical values for future use
        if i == 0:
            t_tot = len(xis)
            counter = 0
            new = []
            x_0 = 1e-50
            while counter < t_tot:
                b = 1 - ((1/6)*(x_0**2))
                new.append(b)
                counter += 1
                x_0 += t_step
            c, y_se = comp(xis, new)
            real, y_rse = comp(xis, thetas)
            error = abs((real - c)/real) * 100
            print("Percent error for n=0: " + str(error))
            av_se = (y_se + y_rse) / 2
            se_dif = av_se - y_rse
            print("Standard error average difference for n=0: " + str(se_dif))
            print()
            plt.plot(xis, new, label="Real 0")
        elif i == 1:
            t_tot = len(xis)
            counter = 0
            new = []
            x_0 = 1e-50
            while counter < t_tot:
                b = (mt.sin(x_0))/x_0
                new.append(b)
                counter += 1
                x_0 += t_step
            c, y_se = comp(xis, new)
            # error analysis
            real, y_rse = comp(xis, thetas)
            error = abs((real - c) / real) * 100
            print("Percent error for n=1: " + str(error))
            av_se = (y_se + y_rse)/2
            se_dif = av_se - y_rse
            print("Standard error average difference for n=1: " + str(se_dif))
            print()
            plt.plot(xis, new, label="Real 1")
        elif i == 5:
            t_tot = len(xis)
            counter = 0
            new = []
            x_0 = 1e-50
            while counter < t_tot:
                b = 1/(mt.sqrt(1+(x_0**2)/3))
                new.append(b)
                counter += 1
                x_0 += t_step
            c, y_se = comp(xis, new)
            real, y_rse = comp(xis, thetas)
            error = abs((real - c) / real) * 100
            print("Percent error for n=5: " + str(error))
            av_se = (y_se + y_rse) / 2
            se_dif = av_se - y_rse
            print("Standard error average difference for n=5: " + str(se_dif))
            plt.plot(xis, new, label="Real 5")
        # plot unknown n value answers
        plt.plot(xis, thetas, label=i)

    # set plot details and limits
    plt.title("Lane-Emden Analytical Solutions With Runge-Kutta 4th Order")
    plt.xlim(0, 10)
    plt.axhline(y=0, color="black")
    plt.ylim(-1.1, 1.1)
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\theta$')
    plt.legend()
    plt.show()


# integration def that is called from the main def
def rk_integrate(n, t_step):
    x_0 = 1e-50  # need some small value that is close to 0 but not exactly 0
    y_0 = 1.0  # starting value for the y pos
    z_0 = 0.0  # starting value for the z pos

    # do the integration and compile the results into two lists
    xs = [x_0]
    ys = [y_0]
    while x_0 < 2500:
        x_0, y_0, z_0 = runge_kutta.integrate(x_0, y_0, z_0, n, t_step)
        if type(y_0) == complex:  # complex values of y_0 sometimes occur when n is not an integer
            break
        xs.append(x_0)
        ys.append(y_0)
    return xs, ys


def comp(x, y):
    x_comp = 0
    y_comp = 0
    for i in x:
        x_comp += i
    for i in y:
        y_comp += i
    x_comp = x_comp/(len(x))
    y_comp = y_comp/(len(y))
    c = mt.sqrt(x_comp**2 + y_comp**2)

    y_sd = st.stdev(y)
    y_se = y_sd/mt.sqrt(len(y))
    return c, y_se


if __name__ == '__main__':
    main()
