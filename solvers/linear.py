from pa_instance import PAInstance
from solution import PASolution
from typing import List, Optional
from copy import deepcopy

class LinearSolver:
    """ O(n) solver for the Ping Assignment problem. """

    def __init__(self, instance: PAInstance):
        """ Initialises the solver.
        
            Parameters:
                * instance (PAInstance): Ping Assignment instance to solve.
        """
        self.instance = instance

    def __solve_for_bs(self, bs: List[int], assignment: List[Optional[int]]) -> Optional[int]:
        m = max(bs)

        if m == 0:
            return 0

        all_max = [idx for idx, val in enumerate(bs) if val == m]
        ass_idx = assignment.index(None)
        ass_free_sz = len(assignment) - 2 * ass_idx

        for i, base in enumerate(all_max):
            assert assignment[ass_idx + i] == None
            assignment[ass_idx + i] = base
            bs[base] -= 1

            if m > 1:
                assert assignment[-ass_idx - 1 - i] == None
                assignment[-ass_idx - 1 - i] = base
                bs[base] -= 1

        return (m - 1) * len(all_max) * (ass_free_sz - len(all_max))


    def solve(self) -> PASolution:
        """ Solves the problem using a linear-time algorithm. """
        bs = deepcopy(self.instance.base_size)
        assignment = [None] * self.instance.num_items
        obj = 0

        while (contrib := self.__solve_for_bs(bs=bs, assignment=assignment)) > 0:
            obj += contrib

        return PASolution(
            lower_bound=obj,
            upper_bound=obj,
            assignment=assignment
        )