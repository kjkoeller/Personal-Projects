"""
N-Body problem using the Runge-Kutta 4th order method accounting for eccentricity of each body

Kyle Koeller
Physics 336, Spring 2020
Written 4/14/2020
"""
from vpython import *
import math as m
import matplotlib.pyplot as plt
'''
Left in case a switch to different units occurs
G = 6.7e-11  # Gravitational constant
AU = 1.49597870691e11  # meters
mass_E = 5.9724e24  # kg
radius_E = 6378  # km
'''
G = 4 * (m.pi ** 2)  # this is allowed because we designate the suns mass as 1.0
c = 3e8  # m/s

# manually change the time-steps in the list and the list names for graphing purposes
dt = [0.998/100, 0.998/1000, 0.998/10000]
dt_names = ["1/100", "1/1000", "1/10000"]
dt_counter = 0

for i in dt:
    scene = canvas(width=1500, height=775)
    
    sun = sphere(pos=vector(0, 0, 0), radius=0.05, name="Sun", color=color.yellow,
                 make_trail=True, interval=10, retain=1000, canvas=scene)
    sun.mass = 1
    sun.v = vector(0, 0, 0)

    # earth position is the minimum distance from the Sun, the max is 1 + 0.017
    earth = sphere(pos=vector(1.0 * (1 - 0.017), 0, 0), radius=sun.radius / 109.2, name="Earth", color=color.blue,
                   make_trail=True, interval=10, retain=75, canvas=scene)
    earth.period = 1.0
    earth.mass = sun.mass / 333000
    earth.e = 0.017
    earth.a = 1.0
    # circular orbit speed is 2pi
    # the following equation calculates the max velocity when the distance is at a minimum
    # to find the minimum velocity change the 1 + earth.e to 1 - earth.e and vis versa for the 1 - earth.e
    earth.v_max = m.sqrt(G*sun.mass)*m.sqrt((1+earth.e)/(earth.a*(1-earth.e)))*(1+earth.mass)
    earth.v = vector(0, earth.v_max, 0)
    earth.energy = -earth.mass/m.sqrt(mag2(earth.pos)) + 0.5*earth.mass*(mag2(earth.v))

    venus = sphere(pos=vector(0.72 * (1 - 0.007), 0, 0), radius=earth.radius*0.952, name="Earth", color=color.yellow,
                   make_trail=True, interval=10, retain=50, canvas=scene)
    venus.mass = earth.mass*0.815
    venus.e = 0.007
    venus.a = 0.72
    venus.v_max = m.sqrt(G*sun.mass)*m.sqrt((1+venus.e)/(venus.a*(1-venus.e)))*(1+venus.mass)
    venus.v = vector(0, venus.v_max, 0)

    mars = sphere(pos=vector(1.52 * (1 - 0.093), 0, 0), radius=earth.radius * 0.53, name="Mars", color=color.red,
                  make_trail=True, interval=10, retain=100, canvas=scene)
    mars.mass = earth.mass * 0.107
    mars.e = 0.093
    mars.a = 1.52
    mars.v_max = m.sqrt(G*sun.mass)*m.sqrt((1+mars.e)/(mars.a*(1-mars.e)))*(1+mars.mass)
    mars.v = vector(0, mars.v_max, 0)

    jupiter = sphere(pos=vector(5.2 * (1 - 0.048), 0, 0), radius=earth.radius * 10.517, name="Jupiter",
                     color=color.orange, make_trail=True, interval=10, retain=500, canvas=scene)
    jupiter.mass = sun.mass / 1000
    jupiter.e = 0.048
    jupiter.a = 5.2
    jupiter.v_max = m.sqrt(G*sun.mass)*m.sqrt((1+jupiter.e)/(jupiter.a*(1-jupiter.e)))*(1+jupiter.mass)
    jupiter.v = vector(0, jupiter.v_max, 0)

    saturn = sphere(pos=vector(9.54 * (1 - 0.056), 0, 0), radius=earth.radius * 8.552, name="Jupiter",
                    color=color.green, make_trail=True, interval=10, retain=1000, canvas=scene)
    saturn.mass = earth.mass*95.2
    saturn.e = 0.056
    saturn.a = 9.54
    saturn.v_max = m.sqrt(G*sun.mass)*m.sqrt((1+saturn.e)/(saturn.a*(1-saturn.e)))*(1+saturn.mass)
    saturn.v = vector(0, saturn.v_max, 0)

    # if you add any planets/stars you must add them to the lists below
    bodies = [sun, venus, earth, mars, jupiter, saturn]
    ecc_list = [venus.e, earth.e, mars.e, jupiter.e, saturn.e]
    a_list = [venus.a, earth.a, mars.a, jupiter.a, saturn.a]

    t_stop = 25
    counter = 0

    # make initial lists for while loop
    t_list = []
    x_list = []
    e_list = []
    t = 0
    while t < t_stop:
        # rate(300) # comment this out for dt of 1000th or lower
        # If you comment this out, the rate will run at what the computer max is
        for body in bodies:
            old_pos = body.pos
            v_old = body.v

            # makes sure the counter never goes beyond the length of either the semi-major axis or eccentricity lists
            if counter >= len(ecc_list):
                counter = 0

            # reset the acceleration vectors to 0 each loop through
            a0 = vector(0, 0, 0)
            a1 = vector(0, 0, 0)
            a2 = vector(0, 0, 0)

            # step 1
            # Add up all of the forces exerted on 'body'
            for other in bodies:
                # Don't calculate the body's attraction to itself
                if body is other:
                    continue
                # compute the distance between the 2 bodies
                r = other.pos - old_pos

                # Compute the force of attraction and the direction
                # this equation also takes into account a planets precession around the star
                f_calc1 = (G*other.mass*hat(r))/(mag(r)**2) + \
                          (6*pi*(other.mass+body.mass))*hat(r)/((c**2)*a_list[counter]*(1-ecc_list[counter]**2))
                a0 += f_calc1

            pos_new = old_pos + v_old*(i/2) + a0*0.125*(i**2)

            # step 2
            # Add up all of the forces exerted on 'body'
            for other in bodies:
                # Don't calculate the body's attraction to itself
                if body is other:
                    continue
                # compute the distance between the 2 bodies
                r = other.pos - pos_new

                # Compute the force of attraction and the direction
                # this equation also takes into account a planets precession around the star
                f_calc2 = (G*other.mass*hat(r))/(mag(r)**2) + \
                          (6*pi*(other.mass+body.mass))*hat(r)/((c**2)*a_list[counter]*(1-ecc_list[counter]**2))
                a1 += f_calc2

            pos_new = old_pos + v_old*i + a1*0.5*(i**2)

            # step 3
            # Add up all of the forces exerted on 'body'
            for other in bodies:
                # Don't calculate the body's attraction to itself
                if body is other:
                    continue
                # compute the distance between the 2 bodies
                r = other.pos - pos_new

                # Compute the force of attraction and the direction
                # this equation also takes into account a planets precession around the star
                f_calc3 = (G*other.mass*hat(r))/(mag(r)**2) + \
                          (6*pi*(other.mass+body.mass))*hat(r)/((c**2)*a_list[counter]*(1-ecc_list[counter]*2))
                a2 += f_calc3

            pos = old_pos + v_old*i + ((a0+(a1*2))*((1/6)*(i**2)))
            v = v_old + (a0+(a1*4)+a2)*((1/6)*i)

            # assigns the final variables to the actual planet for plotting in VPython
            body.pos = pos
            body.v = v

            # add 1 to the counter to continue through the semi-major axis and eccentricity lists
            counter += 1

        # calculate the energy at each time step and append the total to a list
        e_pot = -earth.mass / m.sqrt(mag2(earth.pos))
        e_kinetic = 0.5 * earth.mass*(mag2(earth.v))
        e_tot = e_pot + e_kinetic

        # append all variables to lists for further use
        e_list.append(e_tot)
        t_list.append(t)
        x_list.append(earth.pos.x)

        t += i

        scene.follow(sun)  # makes sure the scene stays directed on the sun

    # percent error from initial energy to final
    counter = 0
    e_calc = 0
    # find the total energy over all time and average it it out over all data points
    for j in e_list:
        e_calc += j
        counter += 1
    e_calc = e_calc/counter

    error = abs((earth.energy - e_calc)/earth.energy) * 100

    # print statements are for displaying the initial energy of the system and then the averaged calculated energy
    # and it's % error
    print("Time step is: " + dt_names[dt_counter])
    print("")

    print("Theoretical Energy: " + str(earth.energy) + " J")
    print("Calculated Energy: " + str(e_calc) + " J")
    print("Energy Error: " + str(error))
    print("")

    # plots the energy vs. time graph for each time step
    plt.title("Energy vs. Time")
    plt.plot(t_list, e_list, label="Time-step: " + str(i))
    plt.xlabel("Time (yr)")
    plt.ylabel("Total Energy (J)")
    dt_counter += 1

    # initial variables for period calculation
    prev = x_list[0] / abs(x_list[0])
    counter_pos = 0
    time_s = 0
    start_time = []
    finish_time = []
    counter_main = 0
    skip_counter = 0

    # finds the times for the when omega changes signs and add them to a list
    while counter_pos <= 10:
        for k in x_list:
            # checks to see if it is the first loop, if it is assigns the sign to -1
            # if it not the first run through then it finds whether i is positive or negative
            if k == 0:
                sign = -1
            else:
                # checks for positive or negative value of i
                sign = k / abs(k)
            counter_pos += 1
            # checks to see if the new value is the opposite of the previous stored value
            if sign == -prev:
                prev = sign
                skip_counter += 1
                # skips the first 10 sign changes to pass any initial instability of the planet
                if skip_counter >= 10:
                    # assigns the time positions
                    if counter_main % 2 == 1:
                        time_s = counter_pos
                        counter_main += 1
                    elif counter_main % 2 == 0:
                        time_f = counter_pos
                        counter_main += 1
                        start_time.append(t_list[time_s])
                        finish_time.append(t_list[time_f])

    # calculate periods and put those periods into a list
    time_counter = 0
    calc = []
    while time_counter < len(finish_time):
        time_start = start_time[time_counter]
        time_finish = finish_time[time_counter]

        calc.append((time_finish - time_start) * 2)
        time_counter += 1

    # print out the different periods to take the standard deviation of all periods calculated per time step
    print("")
    print(calc)
    print("")

    # adds all the periods together
    total = 0
    calc_counter = 0
    for time in calc:
        # the if statement skips the first calculated period as that has an inherent error because of the skipping
        # of the first 10 sign changes above
        if calc_counter == 0:
            calc_counter += 1
            continue
        else:
            total += time
            calc_counter += 1

    # averages the period and calls the function for printing
    calc_period = total/(time_counter-1)
    error = abs((earth.period - calc_period)/earth.period) * 100

    # print out each component of the period
    print("Theoretical Period: " + str(earth.period))
    print("Calculated Period: " + str(calc_period))
    print("Period Error: " + str(error))
    print("")

# display the final plot on a logarithmic scale
# having this at the end of the code puts all graphs onto 1 displayed graph
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()
