#! /usr/bin/env python3
import random
import numpy as np
from functools import partial
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Stock:
    """
    This class represents a stock, each field is needed to generate the
    model for the price over time.

    It is made as a dataclass because I wanted to try them out (it is a new
    feature introduced in 3.7) and I figured this would be a nice opportunity
    """
    mu: float = 0.
    sigma: float = field(default_factory=partial(np.random.normal, scale=5e-3))
    dt: float = 1.  # esto representa dias
    x0: float = field(default_factory=partial(np.random.gamma, shape=50.))
    t0: datetime = field(default=datetime(2010, 1, 1))
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    _b: np.ndarray = field(default_factory=lambda: np.array([]))

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
                    size=(2*pos-len(self._b))  # compute some extra
                )
            ))
            step = np.exp((self.mu - self.sigma**2/2) * self.dt) \
                * np.exp(self.sigma * self._b)
            self.prices = self.x0 * step.cumprod()

        return self.x0 if pos == 0 else self.prices[pos-1]


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
        Return the profit of the portfolio between two dates, if annualized
        is True, it returns the annualized value.

        It is important to note that since the timedelta object in python
        has only a resolution of days, some results may be unintuitively a bit
        off.
        In practice, let date1=datetime(2019,1,1) (janaury first, 2019) and
        date2=datetime(2019,7,1) (july first, 2019), that is, one semester
        apart, and let 'p' be the profit between these two dates. One may
        think intuitively that since a year is 2 semesters, the annualized
        profit would be just (1+p)**2 - 1. This does not hold true, because
        timedelta(days=356)/(date2-date1) is not exactly 2, but rather 2.017
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
        """
        I used this function to avoid repeating some code and also it might
        be useful to plot the whole portfolio between two dates
        """
        if not isinstance(date, datetime):
            raise TypeError(
                f"Date should be of type datetime, got {type(date)}"
            )
        return sum(stock.price(date) for stock in self.stocks)


def main():
    """
    Example usage of the classes implemented
    """
    p = Portfolio()
    d1 = datetime(2018, 1, 1)  # january 1st, 2018
    d2 = datetime(2018, 2, 1)  # february 1st, 2018
    profit = p.profit(d1, d2)
    a_profit = p.profit(d1, d2, annualized=True)

    print(f"The profit between {d1} and {d2} is {profit*100:.2f}%")
    print(
        f"The annualized profit between {d1} and {d2} is {a_profit*100:.2f}%"
    )


if __name__ == '__main__':
    main()
