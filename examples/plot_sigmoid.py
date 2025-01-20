import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x, a=-10, b=0., c=-30, d=30):
    y = d/(1+np.exp(a*(x-b))) + c
    return y

def multiplicative_inverse(x, a=-2, b=1, c=1):
    y = a / (x ** b) + c
    return y

def tangent(x, a=10, b=10, c=0, d=0):
    y = -a/(np.tanh(b*x) + c)
    return y

def exp_decay(x):
    y = (np.exp(1/x) - 1)/ (np.exp(1/x +1 ))
    return y

def log(x, a=-2, b=-100, c=50):
    y=a*np.log(1 + np.exp(b*x+c))
    return y



def plot_2_functions():
    x = np.linspace(-1, 5, 100)

    # collision geometry:
    y_sigmoid = sigmoid(x)
    y_multiplicative_inverse = multiplicative_inverse(x) #geometry collision
    y_tangent = tangent(x) #, a=1, b=1, c=1, d=0)
    y_exp_decay = exp_decay(x)
    y_log = log(x)

    # finsler:
    y_sigmoid = sigmoid(x, d=-30, c=30)
    y_multiplicative_inverse = multiplicative_inverse(x, a=1) #geometry collision
    y_tangent = tangent(x) #, a=1, b=1, c=1, d=0)
    y_exp_decay = exp_decay(x)
    y_log = log(x, a=2, b=-100, c=50)

    plt.plot(x, y_sigmoid, 'b-', label='sigmoid')
    plt.plot(x, y_multiplicative_inverse, 'g-', label='multiplicative inverse')
    # plt.plot(x, y_tangent, 'r--', label='tangent function')
    # plt.plot(x, y_exp_decay, 'y-', label='exponential decay')
    plt.plot(x, y_log, 'y-', label='log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_2_functions()