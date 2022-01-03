from pa_instance import PAInstance
from solvers.qap import QAPSolver
from solvers.colgen import ColumnGenerationSolver
from solvers.linear import LinearSolver

if __name__ == '__main__':
    instance = PAInstance.get_random(num_items_hint=12, num_bases=3)

    print('=== INSTANCE ===')
    print(instance)

    solver1 = QAPSolver(instance=instance)
    sol1 = solver1.solve()

    print('=== COMPACT SOLVER SOLUTION ===')
    print(sol1)

    solver2 = ColumnGenerationSolver(instance=instance, initial_sol=sol1)
    sol2 = solver2.solve()

    print('=== COLUMN GENERATION SOLUTION ===')
    print(sol2)

    solver3 = LinearSolver(instance=instance)
    sol3 = solver3.solve()

    print('=== LINEAR TIME ALGORITHM SOLUTION ===')
    print(sol3)
    