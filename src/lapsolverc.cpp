#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "dense_wrap.hpp"

namespace py = pybind11;

const char *doc_dense = R"pbdoc(
Min cost bipartite matching via shortest augmenting paths for dense matrices

This is an O(n^3) implementation of a shortest augmenting path
algorithm for finding min cost perfect matchings in dense
graphs.  In practice, it solves 1000x1000 problems in around 1
second.

    rids, cids = solve_dense(costs)
    total_cost = costs[rids, cids].sum()

Params
------
costs : MxN array
    Array containing costs.

Returns
-------
rids : array
    Array of row ids of matching pairings
cids : array
    Array of column ids of matching pairings

)pbdoc";


PYBIND11_MODULE(lapsolverc, m) {
    m.doc() = R"pbdoc(
        Linear assignment problem solvers using native c-extensions.
    )pbdoc";

    // m.def("solve_dense", solve_dense_wrap<int, py::array::c_style>, py::arg().noconvert());
    // m.def("solve_dense", solve_dense_wrap<double, py::array::c_style>, py::arg().noconvert());
    // m.def("solve_dense", solve_dense_wrap<float, py::array::c_style>, py::arg().noconvert());
    m.def("solve_dense", solve_dense_wrap<double, py::array::c_style | py::array::forcecast>);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
