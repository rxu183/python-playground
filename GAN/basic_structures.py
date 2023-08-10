import numpy as np
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

def jacobian(function, x, steps):
    pass

def function(x):
    return 2*x

def plot():
    space = np.linspace(-10, 10, 100)
    figure = plt.figure("Something of Some Function")
    #figure.suptitle('Relu Function')
    plt.plot(space, derivative(relu, space, 100))
    #figure.add_subplot(111)
    plt.show()
plot()
