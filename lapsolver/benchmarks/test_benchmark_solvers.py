from scipy.optimize import linear_sum_assignment
import numpy as np
import pytest
import importlib
import sys

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

    def run(costs):
        rids, cids = solve_dense(costs)
        return costs[rids, cids].sum()
        
    return run

def load_solver_scipy():
    from scipy.optimize import linear_sum_assignment

    def run(costs):
        rids, cids = linear_sum_assignment(costs)
        return costs[rids, cids].sum()

    return run

def load_solver_munkres():
    from munkres import Munkres, DISALLOWED
    
    def run(costs):
        m = Munkres()
        idx = np.array(m.compute(costs), dtype=int)
        return costs[idx[:,0], idx[:,1]].sum()
    
    return run

def load_solver_lapjv():    
    from lap import lapjv

    def run(costs):
        r = lapjv(costs, return_cost=True, extend_cost=True)
        return r[0]
    
    return run

def load_solver_ortools():
    from ortools.graph import pywrapgraph

    def run(costs):
        f = 1e3
        valid = np.isfinite(costs)
        # A lot of time in ortools is being spent in constructing the graph.
        assignment = pywrapgraph.LinearSumAssignment()
        for r in range(costs.shape[0]):
            for c in range(costs.shape[1]):
                if valid[r,c]:
                    assignment.AddArcWithCost(r, c, int(costs[r,c]*f))
        
        # No error checking for now
        assignment.Solve()
        return assignment.OptimalCost() / f

    return run

def load_solvers():
    loaders = [
        ('lapsolver', load_solver_lapsolver),
        ('lapjv', load_solver_lapjv),
        ('scipy', load_solver_scipy),
        ('munkres', load_solver_munkres),        
        ('ortools', load_solver_ortools),
    ]

    solvers = {}
    for l in loaders:
        try:
            solvers[l[0]] = l[1]()
        except:
            pass
    return solvers


solvers = load_solvers()
sizes = [
    ([10,10], -80040), 
    ([10,5], -80040), 
    ([20,20], -175988), 
    ([50,20], -467118),
    ([50,50], -467118),    
    ([100,100], 0),
    ([200,200], 0),
    ([500,500], 0),
    ([1000,1000], 0),
    ([5000,5000], 0),
]
size_max = [5000,5000]

np.random.seed(123)
icosts = np.random.randint(-1e4, 1e4, size=size_max)

@pytest.mark.benchmark(
    min_time=1,
    min_rounds=2,
    disable_gc=False,
    warmup=True,
    warmup_iterations=1    
)
@pytest.mark.parametrize('solver', solvers.keys())
@pytest.mark.parametrize('scalar', [int, np.float32])
@pytest.mark.parametrize('size,expected', sizes)

def test_benchmark_solver(benchmark, solver, scalar, size, expected):

    exclude_above = {
        'munkres' : 200,
        'scipy' : 1000,
        'ortools' : 5000
    }    

    benchmark.extra_info = {
        'solver': solver,
        'size': size,
        'scalar': str(scalar)
    }

    s = np.array(size)
    if (s > exclude_above.get(solver, sys.maxsize)).any():
        benchmark.extra_info['success'] = False
        return

    costs = icosts[:size[0], :size[1]].astype(scalar).copy()
    r = benchmark(solvers[solver], costs)
    print(solver, size, r)
    if r != expected:
        benchmark.extra_info['success'] = False

# pytest lapsolver -k test_benchmark_solver -v --benchmark-group-by=param:size,param:scalar -s --benchmark-save=bench