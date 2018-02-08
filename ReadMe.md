**py-lapsolver** implements a fast linear sum assignment problem solver for dense matrices. In practice, it solves 5000x5000 problems in around 3 seconds.

### Install

```
pip install lapsolver [--pre]
```

Windows binary wheels are provided for Python 3.5/3.6. Source wheels otherwise.

### Usage

```python
import numpy as np
from lapsolver import solve_dense

costs = np.array([
    [6, 9, 1],
    [10, 3, 2],
    [8, 7, 4.]
], dtype=np.float32)    

rids, cids = solve_dense(costs)

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

rids, cids = solve_dense(costs)

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
Matrix      Solver
3x3         lapsolver         0.000
            scipy             0.000
            ortools           0.000
            munkres           0.000
10x10       lapsolver         0.000
            scipy             0.001
            ortools           0.000
            munkres           0.001
100x100     lapsolver         0.000
            scipy             0.029
            ortools           0.012
            munkres           1.550
200x200     lapsolver         0.002
            scipy             0.246
            ortools           0.050
            munkres          20.061
500x500     lapsolver         0.011
            scipy             5.738
            ortools           0.318
            munkres               -
1000x1000   lapsolver         0.054
            scipy                 -
            ortools           1.302
            munkres               -
5000x5000   lapsolver         2.433
            scipy                 -
            ortools          33.684
            munkres               -
10000x10000 lapsolver        19.877
            scipy                 -
            ortools               -
            munkres               -
```

Benchmark on non-square matrices
```
                 Runtime [sec]
Matrix Solver
3x2    lapsolver         0.000
       scipy             0.001
       ortools           0.000
       munkres               -
10x5   lapsolver         0.000
       scipy             0.000
       ortools           0.000
       munkres               -
100x10 lapsolver         0.002
       scipy             0.000
       ortools           0.003
       munkres               -
200x20 lapsolver         0.010
       scipy             0.000
       ortools           0.014
       munkres               -
500x50 lapsolver         0.130
       scipy             0.011
       ortools           0.150
       munkres               -
2x3    lapsolver         0.000
       scipy             0.000
       ortools           0.001
       munkres               -
5x10   lapsolver         0.000
       scipy             0.001
       ortools           0.000
       munkres               -
10x100 lapsolver         0.000
       scipy             0.001
       ortools           0.002
       munkres               -
20x200 lapsolver         0.001
       scipy             0.002
       ortools           0.005
       munkres               -
50x500 lapsolver         0.004
       scipy             0.014
       ortools           0.034
       munkres               -
```