**py-hungarian-c** implements a fast linear sum assignment problem solver. It uses a C++ based solver from https://github.com/jaehyunp/

### Install

```
pip install fast-hungarian
```

Windows binary wheels are provided for Python 3.5/3.6. Source wheels else.

### Usage

```python
import numpy as np
from fast_hungarian import solve_minimum_cost

costs = np.array([
    [6, 9, 1],
    [10, 3, 2],
    [8, 7, 4.]
], dtype=np.float32)    

rids, cids = solve_minimum_cost(costs)

for r,c in zip(rids, cids):
    print(r,c) # Row/column pairings
"""
0 2
1 1
2 0
"""
```

You may also want to mark certain pairings impossible

```python
# Matrix with non-allowed pairings
costs = np.array([
    [5, 9, np.nan],
    [10, np.nan, 2],
    [8, 7, 4.]]
)

rids, cids = solve_minimum_cost(costs)

for r,c in zip(rids, cids):
    print(r,c) # Row/column pairings
"""
0 0
1 2
2 1
"""
```

### Benchmarks

Comparison below is carried out on dense square/rectangular matrices using tools from [py-motmetrics](https://github.com/cheind/py-motmetrics).

Benchmark on square matrices
```

                         Runtime [sec]
Matrix    Solver
3x3       fast_hungarian         0.000
          scipy                  0.000
          ortools                0.000
          munkres                0.000
10x10     fast_hungarian         0.000
          scipy                  0.001
          ortools                0.000
          munkres                0.001
100x100   fast_hungarian         0.000
          scipy                  0.028
          ortools                0.012
          munkres                1.604
200x200   fast_hungarian         0.001
          scipy                  0.189
          ortools                0.050
          munkres               16.449
500x500   fast_hungarian         0.008
          scipy                  5.025
          ortools                0.320
          munkres                    -
1000x1000 fast_hungarian         0.052
          scipy                      -
          ortools                1.299
          munkres                    -
5000x5000 fast_hungarian         2.383
          scipy                      -
          ortools               33.906
          munkres                    -
```

Benchmark on non-square matrices
```
                         Runtime [sec]
Matrix    Solver
3x3       fast_hungarian         0.000
          scipy                  0.000
          ortools                0.000
          munkres                0.000
10x10     fast_hungarian         0.000
          scipy                  0.001
          ortools                0.000
          munkres                0.001
100x100   fast_hungarian         0.000
          scipy                  0.028
          ortools                0.012
          munkres                1.604
200x200   fast_hungarian         0.001
          scipy                  0.189
          ortools                0.050
          munkres               16.449
500x500   fast_hungarian         0.008
          scipy                  5.025
          ortools                0.320
          munkres                    -
1000x1000 fast_hungarian         0.052
          scipy                      -
          ortools                1.299
          munkres                    -
5000x5000 fast_hungarian         2.383
          scipy                      -
          ortools               33.906
          munkres                    -
```