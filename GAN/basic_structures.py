import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def plot():
    space = np.linspace(-10, 10, 100)
    figure = plt.figure("Relu Function")
    #figure.suptitle('Relu Function')
    plt.plot(space, relu(space))
    #figure.add_subplot(111)
    plt.show()
plot()
