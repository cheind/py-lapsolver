#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

#include "dense.hpp"

namespace py = pybind11;

template<typename T, int ExtraFlags>
py::tuple solve_dense_wrap(py::array_t<T, ExtraFlags> input1) {
    py::print("solve_dense_wrap<T> where T is",
              std::numeric_limits<T>::is_integer ? "int" : "float",
              sizeof(T) * 8);
    auto buf1 = input1.request();

    if (buf1.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    const int nrows = int(buf1.shape[0]);
    const int ncols = int(buf1.shape[1]);    

    if (nrows == 0 || ncols == 0) {
        return py::make_tuple(py::array(), py::array());
    }

    T *data = (T *)buf1.ptr;

    bool any_finite = false;
    T LARGE_COST = T(0);
    for(int i = 0; i < nrows*ncols; ++i) {
        if (std::isfinite((double)data[i])) {
            any_finite = true;
            LARGE_COST = std::max<T>(LARGE_COST, std::abs<T>(data[i]));
        }
    }
         
    if (nrows == 0 || ncols == 0 || !any_finite) {
        return py::make_tuple(py::array(), py::array());
    }

    const int r = std::min<int>(nrows, ncols);
    const int n = std::max<int>(nrows, ncols);
    LARGE_COST = 2 * r * LARGE_COST + 1;
    std::vector<std::vector<T>> costs(n, std::vector<T>(n, LARGE_COST));

    for (int i = 0; i < nrows; i++)
    {   
        T *cptr = data + i*ncols;
        for (int j =0; j < ncols; j++)
        {
            const T c = cptr[j];
            if (std::isfinite((double)c))
                costs[i][j] = c;
        }
    }


    std::vector<int> Lmate, Rmate;
    solve_dense(costs, Lmate, Rmate);

    std::vector<int> rowids, colids;

    for (int i = 0; i < nrows; i++)
    {
        int mate = Lmate[i];
        if (Lmate[i] < ncols && costs[i][mate] != LARGE_COST)
        {
            rowids.push_back(i);
            colids.push_back(mate);
        }
    }

    return py::make_tuple(py::array(rowids.size(), rowids.data()), py::array(colids.size(), colids.data()));
}
