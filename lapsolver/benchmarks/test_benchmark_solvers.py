from scipy.optimize import linear_sum_assignment
import numpy as np
import pytest
import importlib

"""

square_sizes = [(10,10), (100,100)]
tall_sizes = [(10,10), (20,20)]

square_matrices = [('square', np.random.uniform(size=size)) for size in square_sizes]
tall_matrices = [('tall', np.random.uniform(size=size)) for size in tall_sizes]

@pytest.mark.parametrize('data', [m for m in square_matrices if m[1].shape[0] < 5000])
def test_bench_scipy(benchmark, data):

    benchmark.group = '{} - {}x{}'.format(data[0], data[1].shape[0], data[1].shape[1])
    print('running', benchmark.group)
    benchmark(linear_sum_assignment, data[1])
    

@pytest.mark.parametrize('data', [('tall', m) for m in square_matrices])
def test_bench_lapsolver(benchmark, data):

    benchmark.group = '{} - {}x{}'.format(data[0], data[1].shape[0], data[1].shape[1])
    print('running', benchmark.group)
"""

def load_solver_lapsolver():
    from lapsolver import solve_dense
    return solve_dense

def load_solver_scipy():
    from scipy.optimize import linear_sum_assignment
    return linear_sum_assignment

def load_solver_munkres():
    from munkres import Munkres, DISALLOWED
    
    def run(costs):
        m = Munkres()
        indices = np.array(m.compute(costs), dtype=int)
        return indices[:,0], indices[:,1]
    
    return run

def load_solver_lapjv():    
    from lap import lapjv

    def run(costs):
        r = lapjv(costs, return_cost=True)
        return range(costs.shape[0]), r[1]
    
    return run

def load_solvers():
    loaders = [
        ('lapsolver', load_solver_lapsolver),
        #('scipy', load_solver_scipy),
        #('munkres', load_solver_munkres),
        ('lapjv', load_solver_lapjv),
    ]

    solvers = {}
    for l in loaders:
        try:
            solvers[l[0]] = l[1]()
        except:
            pass
    return solvers


solvers = load_solvers()


@pytest.mark.parametrize('solver', solvers.keys())
@pytest.mark.parametrize('size', [[10,10], [100,100], [1000,1000], [5000,5000]])
#@pytest.mark.timeout(20)
def test_benchmark_solver(benchmark, solver, size):

    np.random.seed(123)
    #costs = np.random.randint(-10000, 10000, size=size)
    costs = np.random.uniform(-100, 100, size=size)
    r = benchmark(solvers[solver], costs.copy())
    #print(costs[r[0], r[1]].sum())

