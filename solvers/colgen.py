from __future__ import annotations
from pa_instance import PAInstance
from solution import PASolution
from typing import Iterable, List, Optional, Tuple
from gurobipy import Env, Model, GRB, Column as GRBCol
import numpy as np

class Column:
    """ A column in the linear extended formulation of the Ping Assignment problem. """

    def __init__(self, instance: PAInstance, base: int, positions: Iterable[int]):
        """ Initialises a column for a given base. 
        
            Parameters:
                * instance (PAInstance): Ping Assignment problem instance.
                * base (int): The base associated with the column.
                * positions (Iterable[int]): an iterable containing the positions
                    occupied by the items of the given base.
        """
        self.instance = instance
        self.base = base
        self.positions = list(positions)

        self.base_col = [0] * self.instance.num_bases
        self.base_col[base] = 1

        self.pos_col = [0] * self.instance.num_items
        for pos in positions:
            self.pos_col[pos] = 1

    def constraint_matrix_col(self) -> List[int]:
        """ Return the coefficients of the column in the constraint matrix. 
            This is given by the juxtaposition of two column vectors:
                * The first refers to the base constraints (at most one column
                    selected per each base).
                * The second refers to the position constraints (for each position,
                    select at most one column covering that position).
        """
        return self.base_col + self.pos_col

    def cost(self) -> int:
        """ Cost of column, computed as the sum of distances between
            each pair of items (without repetition). For example, if
            the column corresponds to assignment:
                _ _ 2 _ 4 _ _ _ 8 _
            The cost is:
                |2-4| + |2-8| + |4-8| = 12
        """
        return sum(
            j2 - j1
            for j1 in range(len(self.pos_col) - 1)
            for j2 in range(j1 + 1, len(self.pos_col))
            if self.pos_col[j1] == 1 and self.pos_col[j2] == 1
        )

    def __str__(self) -> str:
        return f"(B={self.base}, POS={self.positions}, COST={self.cost()})"

    def __eq__(self, other: Column) -> bool:
        return self.base == other.base and self.positions == other.positions


class ColumnGenerationSolver:
    """ Solver for the Ping Assignment problem using an extended MIP formulation.
        The solver performs column generation at the root node, solving the
        continuous relaxation of the problem to optimality. It then produces a
        primal solution solving the formulation as a MIP, but only using the
        columns generated during the exploration of the root node.
    """

    def __init__(self, instance: PAInstance, initial_sol: Optional[PASolution]):
        """ Initialises the Column Generation solver.
        
            Parameters:
                * instance (PAInstance): Ping Assignment problem instance.
                * initial_sol (Solution or None): Optional initial solution.
                
            If no initial solution is provided, the class initialises its column
            pool with a trivial solution which first places all elements of the
            first class in the first available positions, then all elements of
            the second class, etc.

            For example, the initial column pool for an instance with 3 bases
            with 3, 4, 3 items would be:
                [ 1 1 1 _ _ _ _ _ _ _ ] Base 1 (first 3 items)
                [ _ _ _ 2 2 2 2 _ _ _ ] Base 2 (next 4 items)
                [ _ _ _ _ _ _ _ 3 3 3 ] Base 3 (next 3 items)
        """
        self.instance = instance
        self.n = self.instance.num_items
        self.m = self.instance.num_bases
        self.a = self.instance.base_size

        if initial_sol:
            self.__init_column_pool_from_solution(initial_sol)
        else:
            self.__init_column_pool()

        self.__build_model()

    def __init_column_pool(self):
        """ Initialises the column pool providing a (likely very bad)
            primal feasible solution.
        """
        self.column_pool = list()
        self.lower_bound = 0

        cur = 0
        for b in range(self.m):
            col = Column(
                instance=self.instance,
                base=b,
                positions=range(cur, cur + self.a[b])
            )

            self.column_pool.append(col)
            self.lower_bound += col.cost()
            cur += self.a[b]

    def __init_column_pool_from_solution(self, solution: PASolution):
        """ Initialises the column pool from a given primal feasible solution. """
        self.column_pool = list()
        self.lower_bound = 0

        for b in range(self.m):
            positions = [idx for idx, base in enumerate(solution.assignment) if base == b]
            col = Column(
                instance=self.instance,
                base=b,
                positions=positions
            )
            self.column_pool.append(col)
            self.lower_bound += col.cost()

    def __build_model(self):
        """ Initialises the model using the columns in the initial column pool.
            It adds the base constraints first, and the position constraints next.
            It marks all variables as continuous.
        """
        self.env = Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()

        self.model = Model(env=self.env)
        self.model.addVars(len(self.column_pool), vtype=GRB.CONTINUOUS, obj=[c.cost() for c in self.column_pool])
        self.model.ModelSense = GRB.MAXIMIZE
        self.model.update()

        cst_matrix_1 = [c.base_col for c in self.column_pool]
        rhs_1 = [1] * self.m

        self.model.addMConstr(np.array(cst_matrix_1).T, self.model.getVars(), '=', rhs_1)
        self.model.update()

        cst_matrix_2 = [c.pos_col for c in self.column_pool]
        rhs_2 = [1] * self.n

        self.model.addMConstr(np.array(cst_matrix_2).T, self.model.getVars(), '<=', rhs_2)
        self.model.update()

    def __get_duals(self) -> Tuple[List[float], List[float]]:
        """ Gets the duals of the base constraints and the position constraints.
            The first ones are called pi and the second ones, mu.
            This method can only be called after the model (with continuous variables)
            has been solved.
        """
        csts = self.model.getConstrs()
        pi = [cst.Pi for cst in csts[:self.m]]
        mu = [cst.Pi for cst in csts[self.m:]]

        return pi, mu

    def __add_columns(self, columns: List[Column]):
        """ Adds a list of columns to the model and to the column pool. """
        for col in columns:
            if col in self.column_pool:
                print(f"\n/ ! \ Degeneracy: column {col} was already in column pool!")

            self.model.addVar(
                obj=col.cost(),
                column=GRBCol(coeffs=col.constraint_matrix_col(),
                              constrs=self.model.getConstrs())
            )
            self.column_pool.append(col)

    def __get_solution(self) -> PASolution:
        """ Gets the final solution. This includes the upper (dual) bound
            from the continuous relaxation. If the relaxation happens to
            provide an integer solution, then this is also a lower (primal)
            bound. Otherwise, it obtains such a primal bound solving the
            model as an integer problem with the current column pool.
        """
        def var_integer(x: float) -> bool:
            return x < 0.0001 or x > 0.9999

        upper_bound = self.model.ObjVal
        sol = [var.x for var in self.model.getVars()]

        if all(var_integer(var) for var in sol):
            lower_bound = upper_bound
        else:
            lower_bound, sol = self.__resolve_as_integer()

        assignment = [0] * self.n
        for val, col in zip(sol, self.column_pool):
            if val > 0.999:
                for j, coef in enumerate(col.pos_col):
                    if coef == 1:
                        assignment[j] = col.base

        return PASolution(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            assignment=assignment
        )

    def __resolve_as_integer(self) -> Tuple[float, List[float]]:
        """ Re-solves the model as an integer programme using the
            current column pool.
        """
        for var in self.model.getVars():
            var.vtype = GRB.BINARY

        self.model.update()
        self.model.setParam(GRB.Param.TimeLimit, 10)
        self.model.optimize()

        lower_bound = self.model.ObjVal
        sol = [var.x for var in self.model.getVars()]

        return lower_bound, sol

    def solve(self) -> PASolution:
        """ Solves the Ping Assignment problem using a column-generation-based
            heuristic.
        """
        while True:
            self.model.optimize()

            if self.model.Status != GRB.OPTIMAL:
                print('ERROR: Could not solve the RMP model to optimality!')
                raise Exception('Could not solve the Restricted Master Problem')

            pi, mu = self.__get_duals()
            new_columns = self.__solve_separation(pi=pi, mu=mu)

            if len(new_columns) == 0:
                break

            self.__add_columns(new_columns)
            print(f"\rColumn pool size: {len(self.column_pool)}", end='')

        print()
        solution = self.__get_solution()

        return solution

    def __del__(self):
        self.model.dispose()
        self.env.dispose()

    def __solve_separation(self, pi: List[float], mu: List[float]) -> List[Column]:
        """ Solves the separation problem finding, for each base, the column which
            maximally violates the dual constraint. To do so, it solves a quadratic
            integer programme using Gurobi.
        """
        columns = list()

        ienv = Env(empty=True)
        ienv.setParam('OutputFlag', 0)
        ienv.start()
        imodel = Model(env=ienv)
        y = imodel.addVars(self.n, vtype=GRB.BINARY, name='y')
        imodel.setObjective(
            sum(
                (j2 - j1) * y[j1] * y[j2]
                for j1 in range(self.n - 1)
                for j2 in range(j1 + 1, self.n)
            ) -
            sum(
                mu[j] * y[j]
                for j in range(self.n)
            ),
            sense=GRB.MAXIMIZE
        )
        # Placeholder 0 RHS, will be changed for each b
        constr = imodel.addConstr((y.sum() == 0), name='placeall')

        for b in range(self.m):
            imodel.setAttr('RHS', [constr], [self.a[b]])
            imodel.optimize()

            if imodel.Status != GRB.OPTIMAL or imodel.ObjVal <= pi[b]:
                continue

            columns.append(Column(
                instance=self.instance,
                base=b,
                positions=[pos for pos in range(self.n) if y[pos].X > 0.5]
            ))

        imodel.dispose()
        ienv.dispose()

        return columns