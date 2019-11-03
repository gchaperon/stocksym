from dataclasses import dataclass, field
from random import random


@dataclass
class Stock:
    mean: float = field(default_factory=random)
    std: float = field(default_factory=random)


def main():
    ...


if __name__ == '__main__':
    main()
