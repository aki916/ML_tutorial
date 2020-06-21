import random
from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy as np

def get_data(box,n,p_lambda):
    rvs = rd.poisson(p_lambda, n)
    return np.concatenate([box,rvs],axis=0)


def visualize(box):
    x = np.arange(len(box))
    plt.bar(x,box)
    plt.show()

def main():
    box = np.array([])
    box = get_data(box,20,5)
    box = get_data(box,20,10)

    visualize(box)
    # breakpoint()





if __name__ == '__main__':
    main()