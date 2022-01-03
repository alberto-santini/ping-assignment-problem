from dataclasses import dataclass
from typing import List

@dataclass
class PASolution:
    """ A solution to the Ping Assignment Problem. """

    """ Lower (primal) bound on the optimal solution value. """
    lower_bound: float

    """ Upper (dual) bound on the optimal solution value. """
    upper_bound: float

    """ Assignment of items to places. The entry at position i
        tells which base occupies place i in the solution.
    """
    assignment: List[int]

    def pct_gap(self) -> float:
        """ Percentage gap between upper and lower bound. """
        if self.upper_bound > 0:
            return 100 * (self.upper_bound - self.lower_bound) / self.upper_bound
        else:
            return 100

    def __str__(self) -> str:
        return f"{self.assignment} - LB: {self.lower_bound} - UB: {self.upper_bound} - Opt gap: {self.pct_gap():6.2f}%"
