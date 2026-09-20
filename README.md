# ABML algorithm

Argument-based machine learning intelligent tutoring system

Create table data in Orange (or in Excel but then you have to manually specify attribute types). Arguments (and other meta attributes) have to be string type. Attribute types can be continous (numerical) or discrete (categorical).

Install Orange3 for work: 
1. pip3 install Orange3
2. Go to backend folder orange3-abml-master in terminal and run command: pip3 install -e .
3. Go to backend folder orange3-evcrules-master in terminal and run command: pip3 install -e .

Run main.py file

## Tested environment

- Python 3.9.6
- numpy 1.26.4
- Orange3 3.38.1
- scikit-learn 1.6.1

Install with: `pip3 install -r requirements.txt`

Also verified with NumPy 2.x (Python 3.12, Orange3 3.40.0)