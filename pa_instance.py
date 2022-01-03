from __future__ import annotations
from typing import List
from numpy.random import normal

class PAInstance:
    """ Number of items (urls in the original problem) to allocate. """
    num_items: int

    """ Number of bases (hostnames in the original problem). """
    num_bases: int

    """ Size of each base. This is a list of length `num_bases`.
        The entry at position i tells the num of items with base i.
    """
    base_size: List[int]

    def __init__(self, num_items: int, base_size: List[int]):
        """ Initialises an instance of the Ping Assignment problem.
        
            Parameters:
                * num_items (int): Number of items.
                * base_size (list of ints): Number of items in each base.
        """
        self.num_items = num_items
        self.num_bases = len(base_size)
        self.base_size = base_size

    @staticmethod
    def get_random(num_items_hint: int, num_bases: int) -> PAInstance:
        """ Generates a random Ping Assignment instance.
            The instance is guaranteed to have exactly `num_bases` bases.
            The number of items in each base is drawn from a normal distribution
            of mean mu = num_items_hint / num_bases and standard deviation
            sigma = mu / 2 (but the minimum number of items in each base is 2).

            Parameters:
                * num_items_hint (int): hint on the number of items in the instance.
                * num_bases (int): number of bases in the instance.
        """
        bs = num_items_hint / num_bases
        base_size = list(normal(loc=bs, scale=bs/2, size=num_bases).clip(min=2).astype(int))
        return PAInstance(num_items=int(sum(base_size)), base_size=base_size)

    def __str__(self) -> str:
        return f"(N={self.num_items}, BASES={self.base_size})"