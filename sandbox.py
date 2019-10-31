#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N = 100


def main():
    x = np.linspace(-1, 1, N)
    y = 0.1 * x
    plt.plot(y)
    plt.plot([np.random.randn() * 0.2 + y[i] for i in range(N)])
    plt.ylim([-1, 1])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
