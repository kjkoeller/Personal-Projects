"""
Author: Kyle Koeller
Created: 7/16/2022
Last Updated: 7/17/2022

Searches through an active report of all sales for a given day and counts the unique keys and then the number of times
that key appears with a value. Since the report reports all values twice, I divide the total number of the value by 2
"""

import pandas as pd
from collections import Counter


def main():
    """
    Reads in the active report and counts the unique items and outputs the total of that item
    and the item name to a file

    :return: nothing
    """
    # import file, might want this to be a user_input
    df = pd.read_csv("active_report_June_14.txt", delimiter="\t", skiprows=5, header=None)
    # list for the item names
    item_names = list(df[2])

    # creates a dictionary for the item names (key) and the total of that item in the list (value)
    count = Counter(item_names)

    # opens a file with this name to begin writing to the file but makes sure it has a .txt ending for easy reading post
    output_test = None
    while not output_test:
        output_file = input("What is the output file name for the regression tables (either .txt or .tex): ")
        if output_file.endswith(".txt"):
            output_test = True
        else:
            print("This is not an allowed file output. Please make sure the file has the extension .txt or .tex.")
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
