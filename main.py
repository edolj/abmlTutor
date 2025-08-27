#!/usr/bin/env python
# -*- coding: utf-8 -*- 
__author__ = 'edo'

import os
from Orange.data import Table, Domain, StringVariable
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
    arguments_var = next((meta for meta in learning_data.domain.metas if meta.name == "Arguments"), None)

    if arguments_var is None:
        print("Error: 'Arguments' meta attribute not found.")
        return False
    
    old_val = learning_data[row_index][arguments_var]
    if old_val in (None, ""):
        learning_data[row_index][arguments_var] = user_argument
    else:
        learning_data[row_index][arguments_var] = f"{old_val}, {user_argument}"

    return True

def removeArgument(learning_data, row_index, formatedArg):
    arguments_var = next((meta for meta in learning_data.domain.metas if meta.name == "Arguments"), None)
    if arguments_var is None:
        print("Error: 'Arguments' meta attribute not found.")
        return False

    old_val = learning_data[row_index][arguments_var]
    if old_val is None:
        return True

    old_val_str = str(old_val)
    args_list = [arg.strip() for arg in old_val_str.split(",")]
    if formatedArg in args_list:
        args_list.remove(formatedArg)

    new_val = ", ".join(args_list) if args_list else ""
    learning_data[row_index][arguments_var] = new_val

    return True

def add_arguments_meta_column(data: Table) -> Table:
    if any(m.name == "Arguments" for m in data.domain.metas):
        return data

    arguments_var = StringVariable("Arguments")
    new_domain = Domain(
        data.domain.attributes,
        data.domain.class_var,
        metas=[arguments_var] + list(data.domain.metas)
    )

    new_data = Table.from_table(new_domain, data)
    new_data[:, arguments_var] = [[""] for _ in range(len(new_data))]

    return new_data

def main():
    """
    Main function, which contains the ABML interactive loop.
    """

    path = os.getcwd() + "/backend/orange3_abml_master/orangecontrib/abml/data/"
    file_path = path + "bonitete_tutor.tab"

    table = Table(file_path)
    learning_data = add_arguments_meta_column(table)
    learner = abrules.ABRuleLearner()
    
    # optional
    #learner.rule_finder.general_validator.max_rule_length = 3
    #learner.rule_finder.general_validator.min_covered_examples = 5

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
        
        # Extract the critical example from the original dataset
        critical_instances = learning_data[crit_ind]

        # show user 5 critical examples and let him choose
        for index, instance in enumerate(critical_instances[:5]):
            print("(%d) -> %s |||| %s |||| %s" % (index + 1, instance["credit.score"], instance["activity.ime"], problematic[:5][index]))
            # problematic_rules tell us which rules classified wrong e.g. credit score is A but rule classified it as E
            #for pravilo in problematic_rules[index]:
            #    print(pravilo)
            #print()
        
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
            counters, counters_vals, arg_rule, best_rule  = argumentation.analyze_argument(learner, learning_data, critical_index)
            
            # rule is the argumented one by user, best rule provide possible improved rule
            arg_m_score = learner.evaluator_norm.evaluate_rule(arg_rule)
            print("Arg rule: ", arg_rule)
            print("Arg rule m-score: ", arg_m_score)

            # get m-score for best_rule, best rule can be used to improve argument learning
            m_score = learner.evaluator_norm.evaluate_rule(best_rule)
            print("Best rule: ", best_rule)
            print("Best rule m-score: ", m_score)

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
                removeArgument(learning_data, critical_index, formatedArg)
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