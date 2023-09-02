import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random

device = ( #Check if GPU Cuda is available or TPU, otherwise run on CPU.
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#Steps for Performing even a Linear Regression:
#1. Read in/Clean the data:
def read(is_manual, data_loc):
    """
    Input: Implicit Boolean Flag Manual that determines ewhether you want to enter points manually, or read from file.
    Output: A formatted dataset of vectors: A vector of X's and a vector of Y's.
    """
    #How to code a reader please. 
    x = []
    y = []
    if(is_manual): #This determines wehther or not we want to read input from the console or via the file.
        input1 = "c"
        while input1[-1] == 'c':
            input1 = input() #
            if input1[-2] == 'x':
                x.append(float(input1[0:-2]))
            if input1[-2] == 'y':
                y.append(float(input1[0:-2]))
        print(x)
        print(y)
        x = np.array(x)
        y = np.array(y)
    else:
        #It's automatic: Read in data:
        df = pd.read_csv(data_loc, usecols=('Temperature', 'Salinity'))
        df=df.dropna(subset=['Temperature','Salinity'])
        #pd.get_dummies: Converts categorical variables into 0/1 combinations for as many types there are? E.g. 
        x = df["Temperature"].to_numpy() #: filters us the rows that we want to use.
        y = df["Salinity"].to_numpy() #: filters us the rows that we want to use.
    return x, y

#1.5: Create a miniature helper function to help with plotting based on input
def linear(m, b, x):
    return m*x + b

#2: Create a method for calculating the optimal coefficients
def leastSquares(x, y):
    """
    Inputs: numpy arrays x, y that denote the independent and dependent variables respectively:
    Outputs: A two-element list that contains the values of the optimal coefficients that minimize the function via the normal equations.
    """
    #First, we need to create the 2D matrix with the first column being the x-values and the second one being an equivalent length of 1s
    ones = np.arange(len(x), dtype=int)
    ones =np.full_like(ones,1)
    #A = np.concatenate((x, ones),axis=1) #This should theoretically combine by columns.
    A = np.column_stack((x, ones)) #This should combine the two arrays element by element.
    print(A)
    print(y)
    A_T = np.transpose(A)
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_T, A)), A_T), np.transpose(y)) #(A_TA)^-1 A_Ty
    #Randomly find XX points to plot:
    combined = list(zip(x, y))
    random.shuffle(combined)
    points = random.sample(combined, min(50, len(x)))
    res1, res2 = zip(*points) #How does this work !"!??:!?/! I honestly have no clue.
    space = np.linspace(0, 25, 100)
    fig = plt.figure("Salinity-Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Salinity (g/kg)")
    plt.plot(space, linear(theta[0], theta[1], space))
    plt.plot(res1, res2, 'o', color='black')

    plt.show()
    return theta


#2. Create the model: 
# In this case, we'll take two approaches: Normal Equations and directly solving for the values:
# And, trying out a neural network with SGD, just to show that it will converge(hopefully): to the same points:
# So, our Normal Equation formula is as follows:
#
def model():
    x = []
    for type in x:
        return 0
    
#3. Run your model:
def main():
    is_manual = False
    data_loc = "salinity.csv"
    xs, ys = read(is_manual, data_loc)
    coeffs = leastSquares(xs, ys)
    print("Our calculated coefficients were: ", coeffs[0], coeffs[1])
main()