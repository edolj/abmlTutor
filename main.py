#!/usr/bin/env python
# -*- coding: utf-8 -*- 
__author__ = 'edo'

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

NOTACTIVE = {"id", "activity.id", "activity.ime", "Arguments"}

EXPERT = [
    "total.oper.liabilities/assets", "current.ratio", "lt.sales.growth",
    "st.sales.growth", "lt.ebit.margin.change", "net.debt/EBITDA",
    "equity.ratio", "TIE", "ROA",
]

def activate_attribute(data, var):
    if var in data.domain.attributes:
        return data
    new_domain = Domain(
        list(data.domain.attributes) + [var],
        data.domain.class_var,
        metas=[m for m in data.domain.metas if m != var],
    )
    return Table.from_table(new_domain, data)

def addArgument(learning_data, row_index, user_argument):
    arguments_var = next((meta for meta in learning_data.domain.metas if meta.name == "Arguments"), None)

    if arguments_var is None:
        print("Error: 'Arguments' meta attribute not found.")
        return False
    
    old_val = learning_data[row_index][arguments_var]
    if str(old_val).strip() in ("", "?", "nan", "None"):
        learning_data[row_index][arguments_var] = user_argument
    else:
        learning_data[row_index][arguments_var] = f"{old_val},{user_argument}"

    return True

def removeArgument(learning_data, row_index, formatedArg):
    arguments_var = next((meta for meta in learning_data.domain.metas if meta.name == "Arguments"), None)
    if arguments_var is None:
        print("Error: 'Arguments' meta attribute not found.")
        return False
    
    old_val = learning_data[row_index][arguments_var]
    if not old_val:
        return True

    old_val_str = str(old_val).strip()
    groups = []
    temp = ""
    brace_level = 0
    for ch in old_val_str:
        if ch == "{":
            brace_level += 1
        if brace_level > 0:
            temp += ch
        if ch == "}":
            brace_level -= 1
            if brace_level == 0:
                groups.append(temp)
                temp = ""

    groups = [g for g in groups if g != formatedArg]

    new_val = ",".join(groups)
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

def move_to_meta(data, names):
    to_meta = [a for a in data.domain.attributes if a.name in names]
    new_domain = Domain(
        [a for a in data.domain.attributes if a.name not in names],
        data.domain.class_var,
        metas=list(data.domain.metas) + to_meta,
    )
    return Table.from_table(new_domain, data)

def load_learn_test(path, file_name="bonitete_tutor", test_size=0.3, seed=0):
    table = Table(path + file_name + ".tab")
    idx = list(range(len(table)))
    learn_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=table.Y, random_state=seed)
    learn = move_to_meta(table[learn_idx], EXPERT)
    test = table[test_idx]
    return learn, test

def evaluate_on_test(classifier, learning_data, test_data):
    test = test_data.transform(learning_data.domain)
    probs = classifier(test, 1)
    y = test.Y.astype(int)
    ca = float(np.mean(np.argmax(probs, axis=1) == y))
    onehot = np.eye(probs.shape[1])[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    auc = float(roc_auc_score(y, probs[:, 1])) if probs.shape[1] == 2 else None
    return {"CA": ca, "AUC": auc, "Brier": brier, "rules": len(classifier.rule_list)}

def main():
    """
    Main function, which contains the ABML interactive loop.
    """

    path = os.getcwd() + "/backend/orange3_abml_master/orangecontrib/abml/data/"
    learn_table, test_data = load_learn_test(path)
    print(f"Learning: {len(learn_table)}  Test: {len(test_data)}")

    learning_data = add_arguments_meta_column(learn_table)
    learner = abrules.ABRuleLearner(evc=False, m=2, parent_alpha=0.05)
    iteration = 0
    
    # optional
    #learner.rule_finder.general_validator.max_rule_length = 3
    #learner.rule_finder.general_validator.min_covered_examples = 5

    input("Ready to learn? Press enter")

    # MAIN LOOP
    while True:
        stars_with_header("Learning rules...")

        classifier = learner(learning_data)

        # print learned rules
        for rule in classifier.rule_list:
            # distribution of samples to target class, rule, quality
            print(rule.curr_class_dist.tolist(), rule, rule.quality)
        print()

        # ===== measures on test dataset =====
        scores = evaluate_on_test(classifier, learning_data, test_data)
        auc_txt = f"{scores['AUC']:.3f}" if scores["AUC"] is not None else "-"
        print(f"[TEST] iter {iteration}: CA={scores['CA']:.3f} "
              f"AUC={auc_txt} Brier={scores['Brier']:.3f} rules={scores['rules']}")
        print()
        # =====================================

        stars_with_header("Finding critical examples...")
        crit_ind, problematic, problematic_rules = argumentation.find_critical(learner, learning_data)

        # Extract critical example from the original dataset
        critical_instances = learning_data[crit_ind][:5]

        # show user 5 critical examples and let him choose
        for index, instance in enumerate(critical_instances):
            print("(%d) -> %s" % (index + 1, instance))
            #print("(%d) -> %s |||| %s |||| %s" % (index + 1, instance["credit.score"], instance["activity.ime"], problematic[:5][index]))
            # problematic_rules tell us which rules classified wrong e.g. credit score is A but rule classified it as E
            #for pravilo in problematic_rules[index]:
            #    print(pravilo)
        
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

            # store all arguments
            user_arguments = []

            while True:
                user_argument = input("Enter argument (or type 'e' to stop with arguments): ")

                if user_argument.lower() == "e":
                    break

                if user_argument in learning_data.domain:
                    var = learning_data.domain[user_argument]
                    if var.name in NOTACTIVE:
                        print("Wrong argument. Try again.")
                        continue
                    if var in learning_data.domain.metas:
                        learning_data = activate_attribute(learning_data, var)
                        print(f"Attribute {var.name} activated.")
                    
                    getIndex = learning_data.domain.index(user_argument)
                    attribute = learning_data.domain[getIndex]

                    if attribute.is_continuous:
                        while True:
                            sign = input("Enter <= or >= or < or >: ")
                            if sign in (">=", "<=", "<", ">"):
                                user_argument += sign
                                break

                        # Optional if user want to set upper/lower bound
                        while True:
                            bound = input("Enter bound (press Enter to skip): ")
                            if bound == "":
                                break

                            try:
                                float(bound)  # Check if valid number
                                user_argument += bound
                                break
                            except ValueError:
                                print("Please enter a valid number.")

                    # print("user argument:", user_argument)
                    user_arguments.append(user_argument)
                else:
                    print("Wrong argument. Try again.")

            if len(user_arguments) == 1:
                user_argument = user_arguments[0]
            else:
                user_argument = ",".join(user_arguments)

            # change it to format {argument}
            formatedArg = "{{{}}}".format(user_argument)
            # add argument to "Arguments" column in row critical_index
            addArgument(learning_data, critical_index, formatedArg)

            # use calculate_evds for extra precision
            # learner.calculate_evds(learning_data)

            input("Press enter for argument analysis")

            stars_with_header("Analysing argument...")
            counters, arg_rule, best_rule = argumentation.analyze_argument(learner, learning_data, critical_index, user_argument)
            
            if arg_rule is not None:
                # rule is the argumented one by user, best rule provide possible improved rule
                arg_m_score = learner.evaluator_norm.evaluate_rule(arg_rule)
                print("Arg rule: ", arg_rule)
                print("Arg rule m-score: ", arg_m_score)

            if best_rule is not None:
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
        iteration += 1

    return 0


if __name__ == '__main__':
    status = main()