"""
Author: Kyle Koeller
Created: 7/16/2022
Last Updated: 7/17/2022

Searches through an active report of all sales for a given day and counts the unique keys and then the number of times
that key appears with a value. Since the report reports all values twice, I divide the total number of the value by 2
"""

# import packages required
import pandas as pd
from collections import Counter
from os import path


def main():
    """
    Reads in the active report and counts the unique items and outputs the total of that item
    and the item name to a file

    :return: nothing
    """

    # allows for any number of days to be added together
    n = int(input("How many days are there: "))
    count = {}
    for i in range(n):
        while True:
            # makes sure the file pathway is real and points to some file
            # (does not check if that file is the correct one though)
            try:
                # an example pathway for the files
                # C:\Users\Kyle\OneDrive\Computer-Science\Personal_Projects\BFB_Stats\report.txt
                file = input("Enter file path %d: " % (i + 1))
                if path.exists(file):
                    break
                else:
                    continue
            except FileNotFoundError:
                print("Please enter a correct file path")
        # import file, might want this to be a user_input
        df = pd.read_csv(file, delimiter="\t", skiprows=5, header=None)
        # list for the item names
        item_names = list(df[2])

        # creates a dictionary for the item names (key) and the total of that item in the list (value)
        count = Counter(count) + Counter(item_names)

    # opens a file with this name to begin writing to the file but makes sure it has a .txt ending for easy reading post
    output_test = None
    while not output_test:
        output_file = input("What is the output file name for the stats to go to (ends with .txt): ")
        if output_file.endswith(".txt"):
            output_test = True
        else:
            print("This is not an allowed file output. Please make sure the file has the extension .txt")
            print()

    # noinspection PyUnboundLocalVariable
    with open(output_file, 'w') as f:
        # checks to see if the key is labeled as Charge Name and if so, ignores it
        for key in count.keys():
            if key == "Charge Name":
                continue
            else:
                # writes to the file the key, and it's value divided by 2
                f.write("%s, %s\n" % (key, count[key]/2))


if __name__ == '__main__':
    main()
