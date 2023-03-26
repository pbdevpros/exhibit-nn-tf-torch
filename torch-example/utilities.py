#! env python

import numpy as np

def plot_metric(x, y, title, legends):
    import matplotlib.pyplot as plt
    if type(x) != type([]): x = [x]
    if type(y) != type([]): y = [y]
    assert(len(x) == len(y))
    colors = [ 'g*', 'r-', 'bk']
    for i in range(len(x)):
        plt.plot(x[i], y[i], colors[i % len(colors)])
    plt.legend(legends, loc='upper right')
    plt.title(title)
    plt.rcParams["figure.figsize"] = (30, 30)
    plt.grid()
    plt.rc({'font.size': 42})
    plt.show()