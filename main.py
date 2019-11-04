#! /usr/bin/env python3

# from random import random
from dataclasses import dataclass, field
# from datetime import timedelta, datetime
from datetime import datetime, timedelta
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
    mu: float = 0.
    sigma: float = field(default_factory=partial(np.random.normal, scale=5e-3))
    dt: float = 1.  # esto representa dias
    x0: float = field(default_factory=partial(np.random.gamma, shape=50.))
    t0: datetime = field(default=datetime(2010, 1, 1))
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
        if not isinstance(date, datetime):
            raise TypeError(
                f"Date should be of type datetime, got {type(date)}"
            )
        if date < self.t0:
            raise Exception(f"Date shouldn't be less than {self.t0}")

        pos = int((date - self.t0) / timedelta(days=self.dt))
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
        dt = timedelta(days=self.dt)
        n = int((date2 - date1) / dt)

        # print(n)
        dates = (date1 + dt * i for i in range(n))
        return np.array([self.price(date) for date in dates])


class Portfolio:
    def __init__(self, stocks=None, **stock_kwargs):
        if stocks is None:
            self.stocks = [Stock(**stock_kwargs) for i in range(random.randrange(5, 10))]
        else:
            if not all(isinstance(stock, Stock) for stock in stocks):
                raise Exception("All stocks should be of type Stock")
            self.stocks = stocks

    def profit(self, date1, date2, annualized=False):
        """
        Return the profit of the portfolio between two dates
        """
        if date2 < date1:
            raise Exception("The second date must come after the first")

        value1 = self.price(date1)
        value2 = self.price(date2)
        overall = (value2 - value1) / value1
        if annualized:
            year = timedelta(days=365)
            dt = date2 - date1
            return (1 + overall) ** (year/dt) - 1
        return overall

    def price(self, date):
        if not isinstance(date, datetime):
            raise TypeError(
                f"Date should be of type datetime, got {type(date)}"
            )
        return sum(stock.price(date) for stock in self.stocks)

    def _show_graph(self, date1, date2):
        """
        TODO: esto tiene un bug, si los dt de los stocks son distintos se cae
        """
        # ipdb.set_trace()
        p = self.stocks[0].prices_range(date1, date2)
        for stock in self.stocks[1:]:
            p += stock.prices_range(date1, date2)
        plt.plot(p)
        plt.show()


def main():
    # s = Stock()
    # tic = time.time()
    # print(s.price(datetime(2010, 1, 2)))
    # toc = time.time()
    # print(f"primera ejecucion {toc-tic}s")
    # s._show_graph()

    # times = []
    # for i in range(20):
    #     tic = time.time()
    #     s.price(datetime(random.randrange(2010, 2100), 1, 1))
    #     toc = time.time()
    #     times.append(toc-tic)
    # print(f"promedio de 20 ejecuciones posteriores {np.mean(times)}s")

    # s.price(datetime(2010, 4, 1))
    # s._show_graph()

    p = Portfolio(dt=1)
    d1 = datetime(2018, 1, 1)
    d2 = datetime(2018, 3, 1)
    print(f"ganancia de {p.profit(d1, d2)*100:.2f}%")
    print(f"annualized {p.profit(d1, d2, annualized=True)*100:.2f}%")
    p._show_graph(d1, d2)


if __name__ == '__main__':
    main()
