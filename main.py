#!/usr/bin/env python
# -*- coding: utf-8 -*- 

__author__ = 'edo'
import logging, os
from Orange.data import Table
from orangecontrib.evcrules.rules import RulesStar
from Orange.classification.rules import Rule, Selector
from backend.orange3_abml_master.orangecontrib.abml import abrules, argumentation

def stars_with_header(msg):
    """
    Util function for prettier output.
    """
    stars = 70
    txt = "*" * stars + "\n"
    txt += msg
    print(txt)
    logging.info(txt)
    return

def addArgumentToColumn(file_path, row_index, argument_to_add):
    # Read the contents of the .tab file
    with open(file_path, "r") as file:
        rows = file.readlines()

    # Check if the row index is within the range of rows
    if 0 <= row_index < len(rows):
        # Split the row into columns using tab as delimiter
        columns = rows[row_index].rstrip().split('\t')

        # Add the string to the last column
        columns.append(argument_to_add)

        # Join the columns back into a row with tabs as delimiter
        updated_row = '\t'.join(columns) + '\n'

        # Update the specific row in the rows list
        rows[row_index] = updated_row

        # Write the updated contents back to the .tab file
        with open(file_path, "w") as file:
            file.writelines(rows)
    else:
        print("Row index out of range.")

def main():
    """
    Main function, which contains the ABML interactive loop.
    """

    path = os.getcwd() + "/backend/orange3_abml_master/orangecontrib/abml/data/"
    file_path = path + "zoo.tab"

    input("Ready to learn? Press enter")

    # MAIN LOOP
    while True:
        # learn
        stars_with_header("Learning rules...")
        learning_data = Table(file_path)
        learner = abrules.ABRuleLearner()
        learner.calculate_evds(learning_data)
        classifier = learner(learning_data)

        for rule in classifier.rule_list:
            print(rule.curr_class_dist.tolist(), rule, rule.quality)
        print()

        # critical examples
        stars_with_header("Finding critical examples...")
        crit_ind, problematic, problematic_rules = argumentation.find_critical(learner, learning_data)

        # Extract the critical instances from the original dataset
        critical_instances = learning_data[crit_ind]

        most_critical_index = crit_ind[0]
        print("Most critical index: ", most_critical_index)

        # input arguments
        while True:
            # input argument
            stars_with_header("Argument input...")
            # take input as argument
            user_argument = input("Enter argument: ")
            # change it to format {argument}
            formatedArg = "{{{}}}".format(user_argument)
            #print(formatedArg)
            # add argument to argument column in row most_critical_index
            addArgumentToColumn(file_path, most_critical_index + 3, formatedArg)
            learning_data = Table(file_path)
            learner = abrules.ABRuleLearner()
            learner.calculate_evds(learning_data)

            input("Press enter for argument analysis")

            # analyse argument
            stars_with_header("Analysing argument...")
            counters, counters_vals, rule, prune = argumentation.analyze_argument(learner, learning_data, most_critical_index)

            while True:
                inp = input("(c)hange argument, (s)tart with blank argument, (d)one with argumentation (new critical): ")
                if inp in ('c', 's', 'd'):
                    break
            if inp == 'd':
                # show which critical example was done in this iteration
                most_critical_instance = learning_data[most_critical_index]
                print("Most critical instance:", most_critical_instance)
                break
            else:
                input_mode = inp

        # increment iteration
        stars_with_header("Next iteration:")

    return 0


if __name__ == '__main__':
    status = main()