import numpy as np
import math
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x) #Very simple function!

def sigmoid(x):
    return 1/(1 + np.exp(-x)) #Fancy.

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.maximum(0.01*x, x)

def derivative(function, x, steps): #Assume: lim delta_x --> 0, then derivative = func(x + delta_x) - func(x) / delta_x. 
    delta_x = 1/steps
    return (function(x + delta_x) - function(x)) / delta_x 

def integral(function, x, steps): #Again, we can approximate this with tiny rectangles that are the height of the function, but have width delta_x.
    delta_x = 1/steps
    #Let's say we have a vector - 
    #function(x) returns the function applied on every element.
    #We want the sum of those products, so we might just a dot product?
    result = np.array([0]) #This is C/initial value
    for i, element in enumerate(function(x) * delta_x):
        result = np.append(result, element + result[i])
    return result[1:]

def exponential(b, m, x):
    return (math.exp(b))*pow(x, m)

def jacobian(function, x, steps):
    #What is a jacobian: 
    pass

def numpyPlayground(x):
    np.transpose(x) #This tranposes a numpy array along a given axis.
    np.concatenate(x) #This should concatenate two numpy arrays by a given axis
    np.full_like(x) #This creates a copy of a numpy array with the same size.
    #df = pd.read_csv("FILE_NMAE") #This reads in a pandas dataframe.
    #df["Column Name"] is how you access columns when looking for things like that. 
    
def function(x):
    return 2*x

def plot():
    #x = [1, 2, 3]
    #y = [5, 6, 7]
    #temp = list(zip(x,y))
    #print(list(zip(*temp)))
    space = np.linspace(0, 125, 200)
    figure = plt.figure("Graph Output")
    figure.suptitle('Relu Function')
    plt.plot(space, derivative(relu, space, 500))
    #plt.plot(space, relu(space))
    #m = 0.500048861195614
    #b = -2.5767014308705485
    #plt.plot(space, exponential(b,m, space))
    #y = [0.833,0.589]
    #x = [120,60]
    #plt.plot(x, y, 'o', color='black')
    #plt.title("I-V Curve")
    #plt.xlabel("Voltage(Volts)")
    #plt.ylabel("Current(Amps)")

    #figure.add_subplot(111)
    plt.show()
plot()
