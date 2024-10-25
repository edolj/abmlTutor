#!/usr/bin/env python
# -*- coding: utf-8 -*- 
__author__ = 'edo'

import os
from Orange.data import Table, Domain
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

def addArgument(learning_data, row_index, user_argument):
    # Find the index of the "Arguments" column in the metas
    arguments_index = next((i for i, meta in enumerate(learning_data.domain.metas) if meta.name == "Arguments"), None)
    
    if arguments_index is None:
        print("Error: 'Arguments' meta attribute not found.")
        return

    # Update the value in the "Arguments" column for the specified row
    learning_data[row_index].metas[arguments_index] = user_argument

def removeArgument(learning_data, row_index):
    # Find the index of the "Arguments" column in the metas
    arguments_index = next((i for i, meta in enumerate(learning_data.domain.metas) if meta.name == "Arguments"), None)
    
    if arguments_index is None:
        print("Error: 'Arguments' meta attribute not found.")
        return

    # Clear the value in the "Arguments" column for the specified row
    learning_data[row_index].metas[arguments_index] = ''

def main():
    """
    Main function, which contains the ABML interactive loop.
    """

    path = os.getcwd() + "/backend/orange3_abml_master/orangecontrib/abml/data/"
    file_path = path + "bonitete_tutor.tab"

    learning_data = Table(file_path)
    learner = abrules.ABRuleLearner()

    input("Ready to learn? Press enter")

    # MAIN LOOP
    while True:
        stars_with_header("Learning rules...")

        # use calculate_evds for extra precision
        # learner.calculate_evds(learning_data)

        classifier = learner(learning_data)

        # print learned rules
        for rule in classifier.rule_list:
            # distribution of samples to target class, rule, quality
            print(rule.curr_class_dist.tolist(), rule, rule.quality)
        print()

        stars_with_header("Finding critical examples...")
        crit_ind, problematic, problematic_rules = argumentation.find_critical(learner, learning_data)
        #print("Critical index: ", crit_ind, "\nProblematic value: ", problematic)

        # Extract the critical example from the original dataset
        critical_instances = learning_data[crit_ind]

        # show user 5 critical examples and let him choose
        for index, instance in enumerate(critical_instances[:5]):
            print("(%d) -> %s |||| %s |||| %s" % (index + 1, instance["credit.score"], instance["activity.ime"], problematic[:5][index]))
            # problematic_rules tell us which rules classified wrong e.g. credit score is A but rule classified it as E
            for pravilo in problematic_rules[index]:
                print(pravilo)
            print()
        
        while True:
            selectedInstanceIndex = input("Choose critical example (number between 1 and 5): ")
    
            if not selectedInstanceIndex.isdigit() or int(selectedInstanceIndex) not in range(1, 6):
                print("Invalid input. Please choose critical example between 1 and 5.")
                continue
            else:
                break
        
        # find selected example in data, get critical index
        critical_index = crit_ind[:5][int(selectedInstanceIndex) - 1]

        while True:
            stars_with_header("Argument input...")

            while True:
                user_argument = input("Enter argument: ")

                if user_argument in learning_data.domain:
                    break
                else:
                    print("Wrong argument. Try again.")
            
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
            # add argument to "Arguments" column in row critical_index
            addArgument(learning_data, critical_index, formatedArg)

            # use calculate_evds for extra precision
            # learner.calculate_evds(learning_data)

            input("Press enter for argument analysis")

            stars_with_header("Analysing argument...")
            counters, counters_vals, rule, prune, best_rule = argumentation.analyze_argument(learner, learning_data, critical_index)

            # get m-score for best_rule, best rule can be used to improve argument learning
            m_score = learner.evaluator_norm.evaluate_rule(best_rule)
            print("Rule m-score: ", m_score)
            
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
                removeArgument(learning_data, critical_index)
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