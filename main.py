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
# import ipdb


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
    _b: np.ndarray = field(default_factory=lambda: np.array([]))

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

        Lazily computes de values for the prices, which depend on the random
        variables stored in _b
        """
        if not isinstance(date, datetime.datetime):
            raise TypeError(
                f"Date should be of type datetime.datetime, got {type(date)}"
            )
        if date < self.t0:
            raise Exception(f"Date shouldn't be less than {self.t0}")

        pos = int((date - self.t0) / datetime.timedelta(days=self.dt))
        if len(self._b) < pos:
            self._b = np.concatenate((
                self._b,
                np.random.normal(
                    scale=np.sqrt(self.dt),
                    size=(2*pos-len(self._b))  # calcular algunos extra
                )
            ))
            step = np.exp((self.mu - self.sigma**2/2) * self.dt) \
                * np.exp(self.sigma * self._b)
            self.prices = self.x0 * step.cumprod()

        return self.x0 if pos == 0 else self.prices[pos-1]

    def prices_range(self, date1, date2):
        # ipdb.set_trace()
        dt = datetime.timedelta(days=self.dt)
        n = int((date2 - date1) / dt)

        # print(n)
        dates = (date1 + dt * i for i in range(n))
        return np.array([self.price(date) for date in dates])


class Portfolio:
    def __init__(self, stocks=None):
        if stocks is None:
            self.stocks = [Stock() for i in range(random.randrange(5, 10))]
        else:
            if not all(isinstance(stock, Stock) for stock in stocks):
                raise Exception("All stocks should be of type Stock")
            self.stocks = stocks

    def profit(self, date1, date2):
        """
        Return the profit of the portfolio between two dates as a percentage
        """
        if not isinstance(date1, datetime.datetime) or not isinstance(date2, datetime.datetime):
            raise TypeError(
                f"Dates should be of type datetime.datetime"
            )
        value1 = sum(stock.price(date1) for stock in self.stocks)
        value2 = sum(stock.price(date2) for stock in self.stocks)

        return (value2 - value1) / value1 * 100

    def _show_graph(self, date1, date2):
        # ipdb.set_trace()
        p = self.stocks[0].prices_range(date1, date2)
        for stock in self.stocks[1:]:
            p += stock.prices_range(date1, date2)
        plt.plot(p)
        plt.show()


def main():
    # s = Stock()
    # tic = time.time()
    # print(s.price(datetime.datetime(2010, 1, 2)))
    # toc = time.time()
    # print(f"primera ejecucion {toc-tic}s")
    # s._show_graph()

    # times = []
    # for i in range(20):
    #     tic = time.time()
    #     s.price(datetime.datetime(random.randrange(2010, 2100), 1, 1))
    #     toc = time.time()
    #     times.append(toc-tic)
    # print(f"promedio de 20 ejecuciones posteriores {np.mean(times)}s")


    # s.price(datetime.datetime(2010, 4, 1))
    # s._show_graph()

    p = Portfolio()
    d1 = datetime.datetime(2018, 1, 1)
    d2 = datetime.datetime(2019, 1, 1)
    print(f"ganancia de {p.profit(d1, d2)}%")
    p._show_graph(d1, d2)


if __name__ == '__main__':
    main()
