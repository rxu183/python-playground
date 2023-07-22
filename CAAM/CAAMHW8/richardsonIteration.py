import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math

def norm2(error):
    res = 0
    for element in error:
        res += pow(element, 2)
    return math.sqrt(res)

def richardsonAlgo(A, b, x_old, alpha, iterations, xstar):
    figure = plt.figure()
    eigenvalues = la.eigvals(A)
    print(eigenvalues)
    x_old_copy = x_old
    
    error = xstar - x_old
    errors = [norm2(error)]
    iteration = [0]
    I = np.identity(len(A)) #Supposedly returns the first dimension (number of rows ? )
    
    for i in range(iterations):
        x_new = np.add( np.matmul((I - alpha*A), x_old_copy), alpha*b)
        x_old_copy = x_new
        error = xstar - x_new
        errors.append(norm2(error))
        iteration.append(i+1)
        #print(i)

    #print(iteration, errors)
    plt.scatter(iteration, errors)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
    return figure

def main():
    
    A = np.array([[1, -0.5, 0, 0],
         [-0.5, 1, -2/3, 0],
         [0, -2/3, 1, -2/3],
         [0, 0, -0.5, 1]
         ])
    b = np.array([0.5, 0, 0, 0])
    x_old = np.array([0,0,0,0])
    a_1 = 0.5
    a_2 = 0.75
    a_3 = 1
    a_4 = 1.25
    xstar = np.array([2, 3, 3, 1.5])
    iterations = 250

    
    richardsonAlgo(A, b, x_old, a_1, iterations, xstar), 
    richardsonAlgo(A, b, x_old, a_2, iterations, xstar)
    richardsonAlgo(A, b, x_old, a_3, iterations, xstar)
    richardsonAlgo(A, b, x_old, a_4, iterations, xstar)
    
main()

