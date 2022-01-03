from pa_instance import PAInstance
from solution import PASolution
from gurobipy import Env, Model, GRB

class QAPSolver:
    """ Quadrati Integer Optimisation solver for the Ping Assignment
        problem. This solver uses a model which is a special case of
        the Quadratic Assignment Problem.
    """

    def __init__(self, instance: PAInstance, **kwargs):
        """ Initialises the QAP (Quadratic Assignment Problem) solver
            for the Ping Assignment problem.
            
            Parameters:
                * instance (PAInstance): the Ping Assignment problem instance.
                * time_limit (int): Gurobi time limit in seconds. Default: 10.
                * use_ineq (bool): if True, the model uses <= inequalities for
                    both sets of constraints. If False, it uses == equalities.
                    Both models are provably correct.
        """
        self.instance = instance
        self.n = self.instance.num_items
        self.m = self.instance.num_bases
        self.a = self.instance.base_size

        self.time_limit = kwargs.get('time_limit', 10)
        self.use_ineq = kwargs.get('use_ineq', True)

        self.__build_model()

    def __build_model(self):
        """ Builds the Gurobi model, ready to be solved. """
        self.env = Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()

        self.model = Model(env=self.env)
        self.x = self.model.addVars(self.m, self.n, vtype=GRB.BINARY, name='x')

        self.model.setObjective(
            sum(
                (j2 - j1) * self.x[b,j1] * self.x[b,j2]
                for b in range(self.m)
                for j1 in range(self.n - 1)
                for j2 in range(j1, self.n)
            ),
        sense=GRB.MAXIMIZE)

        if self.use_ineq:
            self.model.addConstrs((
                sum(self.x[b,j] for b in range(self.m)) <= 1
                for j in range(self.n)),
            name='allpos')

            self.model.addConstrs((
                sum(self.x[b,j] for j in range(self.n)) <= self.a[b]
                for b in range(self.m)),
            name='allitems')
        else:
            self.model.addConstrs((
                sum(self.x[b,j] for b in range(self.m)) == 1
                for j in range(self.n)),
            name='allpos')

            self.model.addConstrs((
                sum(self.x[b,j] for j in range(self.n)) == self.a[b]
                for b in range(self.m)),
            name='allitems')

        

        self.model.setParam(GRB.Param.TimeLimit, self.time_limit)

    def solve(self) -> PASolution:
        """ Solves the Ping Assignment problem using a quadratic integer
            model which is a special case of the Quadratic Assignment
            Problem's classical formulation.
        """
        self.model.optimize()

        sol = PASolution(upper_bound=self.model.ObjBound,
                       lower_bound=self.model.ObjVal,
                       assignment = [0] * self.instance.num_items)

        for j in range(self.n):
            for b in range(self.m):
                if self.x[b,j].X > 0.5:
                    sol.assignment[j] = b

        return sol

    def __del__(self):
        self.model.dispose()
        self.env.dispose()