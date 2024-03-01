'''
Aim: program to implement basic operations on matrices and vectors
1. create a vector
2. Create a matrix
3. Vector addition, subtraction, multiplication and division
4. matrix operations: add(),multiply(),subtract(),divide(), sqrd(),sum(X axis),transpose arrange shape
'''

import numpy as np

'''
vector_row = np.array([1,2,3])
vector_column = np.array([[1],[2],[3]])
print(vector_row[0])
print(vector_column[0])
A = np.array([[1,2,3],[4,5,6]])
'''

X = np.array([[1,2],[4,5]])
Y = np.array([[7,8],[9,10]])
print("Elemenent wise addition:\n")
print(np.add(X,Y))
print(np.subtract(X,Y))
print(np.multiply(X,Y))
print(np.sqrt(X))

