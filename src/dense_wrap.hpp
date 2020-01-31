#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>

#include "dense.hpp"

namespace py = pybind11;


template<typename T>
std::string numeric_type_name() {
    std::string kind = (std::numeric_limits<T>::is_integer ? "int" : "float");
    auto bytes = sizeof(T);
    return kind + std::to_string(bytes * 8);
}


template<typename T, int ExtraFlags>
py::tuple solve_dense_wrap(py::array_t<T, ExtraFlags> input1) {
    py::print("solve_dense_wrap<T> with T =", numeric_type_name<T>());
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
    py::print("LARGE_COST", LARGE_COST);
    std::vector<std::vector<T>> costs(n, std::vector<T>(n, LARGE_COST));

    for (int i = 0; i < nrows; i++)
    {
        T *cptr = data + i*ncols;
        for (int j =0; j < ncols; j++)
        {
            const T c = cptr[j];
            if (std::isfinite((double)c))
                costs[i][j] = c;
            py::print("i,j,c", i, j, costs[i][j]);
        }
    }


    std::vector<int> Lmate, Rmate;
    solve_dense(costs, Lmate, Rmate);

    for (int i = 0; i < nrows; i++)
    {
        int mate = Lmate[i];
        py::print("i,mate", i, mate);
    }

    std::vector<int32_t> rowids, colids;

    for (int i = 0; i < nrows; i++)
    {
        int mate = Lmate[i];
        if (Lmate[i] < ncols && costs[i][mate] != LARGE_COST)
        {
            rowids.push_back(i);
            colids.push_back(mate);
        }
    }
    py::print("len(rowids)", rowids.size());

    return py::make_tuple(py::array_t<int32_t>(rowids.size(), rowids.data()),
                          py::array_t<int32_t>(colids.size(), colids.data()));
}
