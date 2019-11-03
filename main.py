#! /usr/bin/env python3

# from random import random
from dataclasses import dataclass, field
# from datetime import timedelta, datetime
import datetime
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import time
import random


@dataclass
class Stock:
    """
    Esta clase representa un stock, los campos son utilizados para generar
    el precio del stock en el tiempo
    """
    # mu: float = field(default_factory=partial(np.random.normal, scale=2e-3))
    mu: float = 0.
    sigma: float = field(default_factory=partial(np.random.normal, scale=5e-3))
    # sigma: float = 0.
    dt: float = 1.  # esto representa dias
    x0: float = field(default_factory=partial(np.random.gamma, shape=50.))
    t0: datetime.datetime = field(default=datetime.datetime(2010, 1, 1))
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    _w: np.ndarray = field(default_factory=lambda: np.array([]))

    # def __init__(self, mu=None, sigma=None, **kwargs):
    #     self.mu = mu if mu else np.random.normal(scale=2e-3)
    #     ...

    def _show_graph(self):
        """
        Plots a graph with the current state of the historic prices. Note  that
        the graph shown depends on the last price() execution that added more
        prices.

        I use this method mainly to check how the prices of the stock look
        over time and adjust paramters.
        """
        plt.plot(self.prices)
        plt.show()

    def price(self, date):
        """
        Gets the price of the stock at the given date. The model computes the
        position in the array of prices based of off the initial date of the
        stock.

        Lazily computes de values for the prices, which depend on the
        "random walk" variable _w
        """
        if not isinstance(date, datetime.datetime):
            raise TypeError(
                f"Date should be of type datetime.datetime, got {type(date)}"
            )
        if date < self.t0:
            raise Exception(f"Date shouldn't be less than {self.t0}")

        pos = int((date - self.t0) / datetime.timedelta(days=self.dt))
        if len(self._w) < pos:
            self._w = np.concatenate((
                self._w,
                np.random.normal(
                    scale=np.sqrt(self.dt),
                    size=2*pos  # calcular algunos extra, por que no
                )
            ))
            step = np.exp((self.mu - self.sigma**2/2) * self.dt) \
                * np.exp(self.sigma * self._w)
            self.prices = self.x0 * step.cumprod()
        return self.x0 if pos == 0 else self.prices[pos]


def main():
    s = Stock()
    tic = time.time()
    s.price(datetime.datetime(2011,1,1))
    toc = time.time()
    print(f"primera ejecucion {toc-tic}s")

    times = []
    for i in range(20):
        tic = time.time()
        s.price(datetime.datetime(random.randrange(2010, 2100), 1, 1))
        toc = time.time()
        times.append(toc-tic)
    print(f"promedio de 20 ejecuciones posteriores {np.mean(times)}s")

    s._show_graph()


if __name__ == '__main__':
    main()