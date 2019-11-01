# python code for the plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mu = 1
n = 50
dt = 0.1
x0 = 100
x = pd.DataFrame()
np.random.seed(1)
for sigma in np.arange(0.8, 2, 0.2):
    step = np.exp((mu - sigma**2 / 2) * dt) * np.exp(
        sigma * np.random.normal(0, np.sqrt(dt), (1, n)))
    temp = pd.DataFrame(x0 * step.cumprod())
    x = pd.concat([x, temp], axis=1)
x.columns = np.arange(0.8, 2, 0.2)
plt.plot(x)
plt.legend(x.columns)
plt.xlabel('t')
plt.ylabel('x')
plt.title(
    'Realizations of Geometric Brownian Motion with different variances\n mu=1'
)
plt.show()
