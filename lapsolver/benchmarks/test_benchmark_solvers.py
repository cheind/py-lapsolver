import collections
from scipy.optimize import linear_sum_assignment
import numpy as np
import pytest
import importlib
import sys

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
size_to_expected = collections.OrderedDict([
    ('10x5', -39518.0),
    ('10x10', -80040.0),
    ('20x20', -175988.0),
    ('50x20', -193922.0),
    ('50x50', -467118.0),
    ('100x100', -970558.0),
    ('200x200', -1967491.0),
    ('500x500', -4968156.0),
    ('1000x1000', -9968874.0),
    ('5000x5000', -49969853.0),
])
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
@pytest.mark.parametrize('size', [k for k, v in size_to_expected.items()])

def test_benchmark_solver(benchmark, solver, scalar, size):
    dims = _parse_size(size)
    expected = size_to_expected[size]

    exclude_above = {
        'munkres' : 200,
        'ortools' : 5000
    }

    benchmark.extra_info = {
        'solver': solver,
        'size': size,
        'scalar': str(scalar)
    }

    s = np.array(dims)
    if (s > exclude_above.get(solver, sys.maxsize)).any():
        benchmark.extra_info['success'] = False
        return

    costs = icosts[:dims[0], :dims[1]].astype(scalar).copy()
    r = benchmark(solvers[solver], costs)
    if r != expected:
        benchmark.extra_info['success'] = False


def _parse_size(size_str):
    """Parses a string of the form 'MxN'."""
    m, n = (int(x) for x in size_str.split('x'))
    return m, n


# pytest lapsolver -k test_benchmark_solver -v --benchmark-group-by=param:size,param:scalar -s --benchmark-save=bench
