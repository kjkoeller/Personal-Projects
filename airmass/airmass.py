"""
Find the airmass of numerous targets given their RA and DEC and a locations latitude and longitude

Author: Kyle Koeller
Date Created: March 15, 2019
Last Updated: August 7, 2022
"""

import datetime
import math as mt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysolar.solar import *


def main():
    """
    This is the main function meant to read in the Airmass text file to get the date, time, and targets to be analyzed

    :return: Graphs of the airmass vs. time of day on given date of the year
    """

    file = "Airmass_targets.txt"
    df = pd.read_csv(file, header=None, delim_whitespace=True, skiprows=3, nrows=4)

    # define the date and time zone from the text file read in
    date = df[1][0]
    time_zone = df[1][1]

    # define the solar conversion variable
    solar_conversion = 1.00273791

    # call the GST function and MGST function to gather those values for local times
    gst_1, gst_2 = GST(date)
    mgst, observe_times = MGST(int(time_zone), solar_conversion, gst_1, gst_2)

    # read in the same file as above again to get only the RA and DEC of the target objects
    dh = pd.read_csv(file, delim_whitespace=True, skiprows=10)

    name = list(dh["Name"])
    RA = list(dh["RA"])
    DEC = list(dh["DEC"])
    Lat = df[1][2]
    Long = df[1][3]

    # calls the solar altitude function to get the solar altitude for sunset and astronomical twilight
    # solar_altitude(observe_times, date, Lat, Long)

    # line_style or line_color for either a line plot or a scatter plot respectively
    line_style = [(0, (1, 10)), (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)),
                  (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)),
                  (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
    line_color = ["red", "blue", "orange", "black", "green", "purple", "brown", "black", "grey"]

    # call the function airmass for the length of the 'name' list defined above to get each targets airmass at given
    # local times
    for i in range(0, len(name)):
        air, zdist = airmass(mgst, RA[i], DEC[i], Lat, Long)
        plt.plot(observe_times, air, label=name[i], c=line_color[i], linestyle=line_style[i])
    lower_lim = 0.75
    upper_lim = 2.5

    # produce plot for airmass
    plt.ylim(lower_lim, upper_lim)
    plt.yticks(np.arange(lower_lim, upper_lim + 0.1, 0.25))
    # noinspection PyArgumentList

    # sunset, twilight = height_sun(date, Lat, Long, observe_times)
    twilight, sunset = solar_altitude(observe_times, date, Lat, Long)
    # print(twilight, sunset)

    # noinspection PyArgumentList
    plt.xticks(np.arange(observe_times.min(), observe_times.max(), 1))
    plt.axhline(y=2.0, color='black', linestyle='solid')  # plot a horizontal line at y=2 for airmass limit
    plt.axvline(x=sunset, color="red", linestyle="solid")  # plot a vertical line for sunset time
    plt.axvline(x=twilight, color="purple", linestyle="solid")  # plot a vertical line for astronomical twilight
    plt.text(sunset, 0.8, "Sunset", verticalalignment="center")
    plt.text(twilight, 0.9, "Twilight", verticalalignment="center")
    plt.xlabel("Time of Day (12pm - 12pm)")
    plt.ylabel("Airmass")
    plt.title("Airmasss of Objects on: " + date)
    plt.legend()
    plt.grid()
    plt.show()

    """
    # plot the zenith distance of each object
    for i in range(0, len(name)):
        air, zdist = airmass(mgst, RA[i], DEC[i], Lat, Long)
        plt.scatter(observe_times, zdist, label=name[i], c=line_color[i])
    plt.ylim(0, 90)
    plt.yticks(np.arange(0, 95, 5))
    # noinspection PyArgumentList
    plt.axhline(y=60, color='black', linestyle='solid')
    plt.xticks(np.arange(observe_times.min(), observe_times.max() + 1, 1))
    plt.xlabel("Time of Day (12pm - 12pm)")
    plt.ylabel("Z-Distance (degrees)")
    plt.title("Zenith-Distance of Objects on: " + date)
    plt.legend()
    plt.grid()
    plt.show()
    """


def GST(date):
    """
    Meant to find the Greenwhich Sidereal Time (GST)
    :param date: Time of day
    :return: returns the GST for the day of and the day after
    """
    date_list = date.split("/")
    # number_days = int((275*int(date_list[0]))/9)-(2*int((int(date_list[0])+9)/12))+int(date_list[1])-29

    A = int(int(date_list[2]) / 100)
    B = 2 - A + int(A / 4)

    JD_1 = int(365.25 * (int(date_list[2]) + 4716)) + int(30.6001 * (int(date_list[0]) + 1)) + int(
        date_list[1]) + B - 1524.5
    JD_2 = int(365.25 * (int(date_list[2]) + 4716)) + int(30.6001 * (int(date_list[0]) + 1)) + (
            int(date_list[1]) + 1) + B - 1524.5

    T_factor_1 = (JD_1 - 2451545) / 36525
    T_factor_2 = (JD_2 - 2451545) / 36525

    nutation_long_1, nutation_long_2, true_obliquity_1, true_obliquity_2 = t_obliquity(T_factor_1, T_factor_2)

    c_1 = (-8640184.812866 * T_factor_1) + (0.093104 * (T_factor_1 ** 2)) - (0.0000062 * (T_factor_1 ** 3))
    c_2 = (-8640184.812866 * T_factor_2) + (0.093104 * (T_factor_2 ** 2)) - (0.0000062 * (T_factor_2 ** 3))
    d_1 = (mt.ceil(c_1 / 86400) * 86400) - c_1
    d_2 = (mt.ceil(c_2 / 86400) * 86400) - c_2

    gst_1 = ((6 + ((41 + (50.54841 / 60)) / 60)) + (d_1 / 3600)) + (
            (nutation_long_1 / 15) * mt.cos(mt.radians(true_obliquity_1))) - 24
    gst_2 = ((6 + ((41 + (50.54841 / 60)) / 60)) + (d_2 / 3600)) + (
            (nutation_long_2 / 15) * mt.cos(mt.radians(true_obliquity_2))) - 24

    return gst_1, gst_2


def t_obliquity(T_factor_1, T_factor_2):
    """
    Finds the true obliquity and the long nutation for day 1 and 2
    :param T_factor_1: t factor value for day 1
    :param T_factor_2: t factor value for day 2
    :return: returns the true obliquity and nutation fot day 1 and 2
    """
    ohm_1 = 125.04452 - 1934.136261 * T_factor_1 + 0.0020708 * (T_factor_1 ** 2) + (T_factor_1 ** 3) / 450000
    ohm_2 = 125.04452 - 1934.136261 * T_factor_2 + 0.0020708 * (T_factor_2 ** 2) + (T_factor_2 ** 3) / 450000

    L_1 = (280.4665 / 15) + (36000.7698 / 15) * T_factor_1
    L_2 = (280.4665 / 15) + (36000.7698 / 15) * T_factor_2

    L_prime_1 = (281.3165 / 15) + (481267.8813 / 15) * T_factor_1
    L_prime_2 = (281.3165 / 15) + (481267.8813 / 15) * T_factor_2

    nutation_long_1 = (-17.2 / 3600) * mt.sin(mt.radians(ohm_1)) - (1.31 / 3600) * mt.sin(mt.radians(2 * L_1)) - (
            0.23 / 3600) * mt.sin(mt.radians(2 * L_prime_1)) + (0.21 / 3600) * mt.sin(mt.radians(2 * ohm_1))
    nutation_long_2 = (-17.2 / 3600) * mt.sin(mt.radians(ohm_2)) - (1.31 / 3600) * mt.sin(mt.radians(2 * L_2)) - (
            0.23 / 3600) * mt.sin(mt.radians(2 * L_prime_2)) + (0.21 / 3600) * mt.sin(mt.radians(2 * ohm_2))

    obliquity_1 = (9.2 / 3600) * mt.cos(mt.radians(ohm_1)) + (0.57 / 3600) * mt.cos(mt.radians(2 * L_1)) + (
            0.1 / 3600) * mt.cos(mt.radians(2 * L_prime_1)) - (0.09 / 3600) * mt.cos(mt.radians(2 * ohm_1))
    obliquity_2 = (9.2 / 3600) * mt.cos(mt.radians(ohm_2)) + (0.57 / 3600) * mt.cos(mt.radians(2 * L_2)) + (
            0.1 / 3600) * mt.cos(mt.radians(2 * L_prime_2)) - (0.09 / 3600) * mt.cos(mt.radians(2 * ohm_2))

    mean_obliquity_1 = 23 + ((26 + (21.448 / 60)) / 60) - (46.815 / 3600) * T_factor_1 - (0.00059 / 3600) * (
            T_factor_1 ** 2) + (0.001813 / 3600) * (T_factor_1 ** 3)
    mean_obliquity_2 = 23 + ((26 + (21.448 / 60)) / 60) - (46.815 / 3600) * T_factor_2 - (0.00059 / 3600) * (
            T_factor_2 ** 2) + (0.001813 / 3600) * (T_factor_2 ** 3)

    true_obliquity_1 = obliquity_1 + mean_obliquity_1
    true_obliquity_2 = obliquity_2 + mean_obliquity_2

    return nutation_long_1, nutation_long_2, true_obliquity_1, true_obliquity_2


def MGST(tz, solar_conversion, gst_1, gst_2):
    """
    Finds the mean Greenwhich Sidereal Time (MGST)

    :param tz: time zone
    :param solar_conversion: solar conversion factor
    :param gst_1: Greenwhich sidereal time for day 1
    :param gst_2: Greenwhich sidereal time for day 2
    :return: returns the MGST and the time of day
    """
    observe_times = np.arange(-12, 12.25, 0.1)
    UT = []

    for i in observe_times:
        if (i - tz) > 24:
            UT.append(i - tz - 24)
        else:
            UT.append(i - tz)

    solar_interval = []
    for i in UT:
        solar_interval.append(i * solar_conversion)

    mgst = []
    for i in solar_interval:
        if (gst_1 + i) > 24:
            mgst.append(gst_2 + i)
        else:
            mgst.append(gst_1 + i)

    return mgst, observe_times


def airmass(mgst, ra, dec, lat, long):
    """
    Calculates the airmass of multiple targets

    :param mgst: Mean Greenwhich Sidereal Time
    :param ra: Right Ascension of objects
    :param dec: Declination of objects
    :param lat: Latitude of observer
    :param long: Longtitude of observer
    :return: returns the value of arimass for all objects and the zenith distance
    """
    ra_list = ra.split(":")
    dec_list = dec.split(":")
    lat_list = lat.split(":")
    long_list = long.split(":")

    ra_decimal = float(ra_list[0]) + ((float(ra_list[1]) + (float(ra_list[2]) / 60)) / 60)
    dec_decimal = float(dec_list[0]) + ((float(dec_list[1]) + (float(dec_list[2]) / 60)) / 60)
    lat_decimal = float(lat_list[0]) + ((float(lat_list[1]) + (float(lat_list[2]) / 60)) / 60)
    long_decimal = float(long_list[0]) + ((float(long_list[1]) + (float(long_list[2]) / 60)) / 60)

    lst = []
    for i in mgst:
        if i + (long_decimal / 15) < 24:
            lst.append(i + (long_decimal / 15))
        else:
            lst.append(i + (long_decimal / 15) - 24)

    HA = []
    for i in lst:
        HA.append(((i - ra_decimal) * 15) + 360)

    altitude = []
    az_x = []
    az_y = []
    for i in HA:
        altitude.append(mt.degrees(mt.asin(mt.sin(mt.radians(lat_decimal)) * mt.sin(mt.radians(dec_decimal)) + mt.cos(
            mt.radians(lat_decimal)) * mt.cos(mt.radians(dec_decimal)) * mt.cos(mt.radians(i)))))
        az_x.append(mt.sin(mt.radians(i)))
        az_y.append(mt.cos(mt.radians(i)) * mt.sin(mt.radians(lat_decimal)) - mt.tan(mt.radians(dec_decimal)) * mt.cos(
            mt.radians(lat_decimal)))

    azimuth = []
    counter = 0
    for i in az_x:
        azimuth.append(mt.degrees(mt.atan(i / az_y[counter])))
        counter += 1

    zdist = []
    for i in altitude:
        zdist.append(90 - i)

    air = []
    for i in zdist:
        if 1 / (mt.cos(mt.radians(i))) < 0:
            air.append(100)
        elif 1 / (mt.cos(mt.radians(i))) > 2:
            air.append(1 / (mt.cos(mt.radians(i))))
        else:
            air.append(1 / (mt.cos(mt.radians(i))))

    return air, zdist


def solar_altitude(observe_times, date, lat, long):
    """
    Finds the time at which the sun is in sunset and astronomical twilight

    :param observe_times: time of day with hours and minutes
    :param date: date in the text file
    :param lat: latitude of the observer
    :param long: longitude of the observer
    :return: returns the sunset and twilight times to be plotted
    """
    date_list = date.split("/")

    lat_list = lat.split(":")
    long_list = long.split(":")

    lat_decimal = float(lat_list[0]) + ((float(lat_list[1]) + (float(lat_list[2]) / 60)) / 60)
    long_decimal = float(long_list[0]) + ((float(long_list[1]) + (float(long_list[2]) / 60)) / 60)

    sunset = 0
    twilight = 0

    count = 0
    for i in observe_times:
        if i < 0:
            if int(i) + 23 <= 24:
                d = datetime.datetime(int(date_list[2]), int(date_list[0]), int(date_list[1]), int(i) + 23,
                                      np.absolute(int((i - int(i)) * 60)), 0, 0, tzinfo=datetime.timezone.utc)
            else:
                d = datetime.datetime(int(date_list[2]), int(date_list[0]), int(date_list[1]), -int(i),
                                      np.absolute(int((i - int(i)) * 60)), 0, 0, tzinfo=datetime.timezone.utc)
        else:
            d = datetime.datetime(int(date_list[2]), int(date_list[0]), int(date_list[1]) + 1, int(i),
                                  np.absolute(int((i - int(i)) * 60)), 0, 0, tzinfo=datetime.timezone.utc)
        alt = get_altitude(lat_decimal, long_decimal, d)
        count += 1

        if -2 <= alt <= 2:
            sunset = i
        elif -16 >= alt <= -20:
            twilight = i

    return sunset - 12, twilight - 12


def splitter(a):
    """
    Splits the truncated colon Lat and Long into decimal forms

    :param a: latitude and longitude value
    :return: decimal version of lat and long
    """
    # makes the coordinate string into a decimal number from the text file
    count = 0
    if "-" in a:
        a = a.replace("-", "")
        count = 1
    new = a.split(":")
    num1 = int(new[0])
    num2 = int(new[1])
    num3 = int(float(new[2]))
    b = num1 + ((num2 + (num3 / 60)) / 60)

    if count == 1:
        b = b * -1

    return float(b)


if __name__ == '__main__':
    main()
