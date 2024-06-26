#!/usr/bin/env python
# -*- coding: utf-8 -*- 
__author__ = 'edo'

import os
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
    return

def addArgumentToColumn(file_path, row_index, argument_to_add):
    # Read the contents of the .tab file
    with open(file_path, "r") as file:
        rows = file.readlines()

    # Get the header row and find the index of the 'Arguments' column
    header = rows[0].rstrip().split('\t')
    try:
        arguments_index = header.index("Arguments")
    except ValueError:
        return False  # If 'Arguments' column is not found

    # Check if the row index is within the range of rows
    if 0 <= row_index < len(rows):
        # Split the row into columns using tab as delimiter
        columns = rows[row_index].rstrip().split('\t')

        # Append the argument_to_add value to the "Arguments" column
        columns[arguments_index] += argument_to_add

        # Join the columns back into a row with tabs as delimiter
        updated_row = '\t'.join(columns) + '\n'

        # Update the specific row in the rows list
        rows[row_index] = updated_row

        # Write the updated contents back to the .tab file
        with open(file_path, "w") as file:
            file.writelines(rows)
        return True
    else:
        return False

def removeArgumentFromColumn(file_path, row_index, argument_to_remove):
    # Read the contents of the .tab file
    with open(file_path, "r") as file:
        rows = file.readlines()

    # Check if the row index is within the range of rows (excluding the header row)
    if 0 < row_index < len(rows):
        # Split the header row into columns to find the "Arguments" column index
        header_columns = rows[0].rstrip().split('\t')
        if "Arguments" not in header_columns:
            print("Arguments column not found in the header.")
            return

        arguments_column_index = header_columns.index("Arguments")

        # Split the row into columns using tab as delimiter
        columns = rows[row_index].rstrip().split('\t')

        # Remove the argument_to_remove value from the "Arguments" column
        arguments = columns[arguments_column_index].split(', ')
        if argument_to_remove in arguments:
            arguments.remove(argument_to_remove)
            columns[arguments_column_index] = ', '.join(arguments)
        else:
            print(f"Argument '{argument_to_remove}' not found in the 'Arguments' column.")

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
    file_path = path + "bonitete_tutor.tab"

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
        print("Critical indexes:", crit_ind)
        print("Problematic:", problematic)

        # Extract the critical example from the original dataset
        critical_instances = learning_data[crit_ind]
        print("Critical instances:")
        for index, instance in enumerate(critical_instances[:5]):
            print(index+1, " -> ", instance["credit.score"], " ", instance["activity.ime"], " ", problematic[:5][index])
            
        # show user 5 critical examples and let him choose
        while True:
            selectedInstanceIndex = input("Choose critical example (number between 1 and 5): ")
    
            # Check if the input is not a number or not in the specified range
            if not selectedInstanceIndex.isdigit() or int(selectedInstanceIndex) not in range(1, 6):
                print("Invalid input. Please choose critical instance between 1 and 5.")
                continue
            else:
                break
        
        # selected index is now critical_index
        critical_index = crit_ind[:5][int(selectedInstanceIndex) - 1]

        while True:
            # input argument
            stars_with_header("Argument input...")

            while True:
                # take input as argument
                user_argument = input("Enter argument: ")

                if user_argument in learning_data.domain:
                    break
                else:
                    print("Wrong argument")
            
            getIndex = learning_data.domain.index(user_argument)
            attribute = learning_data.domain[getIndex]
            if attribute.is_continuous:
                while True:
                    sign = input("Enter >= or <= : ")
                    if sign == ">=" or sign == "<=":
                        user_argument += sign
                        break

            # change it to format {argument}
            formatedArg = "{{{}}}".format(user_argument)
            # add argument to argument column in row critical_index
            addArgumentToColumn(file_path, critical_index + 3, formatedArg)
            learning_data = Table(file_path)
            learner = abrules.ABRuleLearner()
            learner.calculate_evds(learning_data)

            input("Press enter for argument analysis")

            stars_with_header("Analysing argument...")
            counters, counters_vals, rule, prune, best_rule = argumentation.analyze_argument(learner, learning_data, critical_index)
            
            if len(counters) > 0:
                counter_examples = learning_data[list(counters)]
                print("Counter examples:")
                for counterEx in counter_examples:
                    print(counterEx)
            else:
                print("No counter examples found for the analyzed example.")

            while True:
                inp = input("(c)hange argument, (d)one with argumentation (new critical): ")
                if inp in ('c', 'd'):
                    break
            if inp == 'c':
                removeArgumentFromColumn(file_path, critical_index + 3, formatedArg)
            if inp == 'd':
                # show which critical example was done in this iteration
                critical_instance = learning_data[critical_index]
                print("This iteration critical example:", critical_instance)
                break

        # increment iteration
        stars_with_header("Next iteration:")

    return 0


if __name__ == '__main__':
    status = main()