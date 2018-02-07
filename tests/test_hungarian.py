import pytest
import numpy as np
import hungarian

def test_small():
    costs = np.array([[6, 9, 1],[10, 3, 2],[8, 7, 4.]], dtype=np.float32)    
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(r, expected)

    costs = np.array([[6, 9, 1],[10, 3, 2],[8, 7, 4.]], dtype=float)    
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(r, expected)

    costs = np.array([[6, 9, 1],[10, 3, 2],[8, 7, 4.]], dtype=int)    
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(r, expected)

def test_plain_array():
    costs = [[6, 9, 1],[10, 3, 2],[8, 7, 4.]]
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(r, expected)

def test_nonsquare():
    costs = np.array([[6, 9],[10, 3],[8, 7]], dtype=float)
    
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1], [0, 1]])
    np.testing.assert_allclose(r, expected)

    r = hungarian.solve_minimum_cost(costs.T) # view test
    expected = np.array([[0, 1], [0, 1]])
    np.testing.assert_allclose(r, expected)

def test_views():
    costs = np.array([[6, 9],[10, 3],[8, 7]], dtype=float)
    np.testing.assert_allclose(hungarian.solve_minimum_cost(costs.T[1:, :]), [[0], [1]])

def test_large():
    costs = np.random.uniform(size=(5000,5000))
    r = hungarian.solve_minimum_cost(costs)

def test_solve_nan():
    costs = np.array([[5, 9, np.nan],[10, np.nan, 2],[8, 7, 4.]])
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(r, expected)

def test_solve_inf():
    costs = np.array([[5, 9, np.inf],[10, np.inf, 2],[8, 7, 4.]])
    r = hungarian.solve_minimum_cost(costs)
    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(r, expected)
    