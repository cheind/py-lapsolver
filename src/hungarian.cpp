#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <limits>

#include "hungarian.hpp"

namespace py = pybind11;

py::tuple solve(py::array_t<double, py::array::c_style | py::array::forcecast> input1) {
    auto buf1 = input1.request();

    if (buf1.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    
    const double INVALID = std::numeric_limits<double>::max();
    
    const int nrows = int(buf1.shape[0]);
    const int ncols = int(buf1.shape[1]);
    
    const int n = std::max(nrows, ncols);

	std::vector<std::vector<double>> costs(n, std::vector<double>(n, INVALID));

    for (int i = 0; i < nrows; i++)
    {   
        double *cptr = (double *)buf1.ptr + i*ncols;
        for (int j =0; j < ncols; j++)
        {
            const double c = cptr[j];
            if (std::isfinite(c))
                costs[i][j] = c;
        }
    }

    std::vector<int> Lmate, Rmate;
    MinCostMatching(costs, Lmate, Rmate);

    std::vector<int> rowids, colids;

    for (int i = 0; i < nrows; i++)
    {
        int mate = Lmate[i];
		if (Lmate[i] < ncols && costs[i][mate] != INVALID)
		{
            rowids.push_back(i);
            colids.push_back(mate);
		}
    }

    return py::make_tuple(py::array(rowids.size(), rowids.data()), py::array(colids.size(), colids.data()));
}



PYBIND11_MODULE(hungarian, m) {
    m.doc() = R"pbdoc(
        A fast hungarian solver based on native c-extensions.
    )pbdoc";

    m.def("solve_minimum_cost", &solve, R"pbdoc(
        Solve it
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}