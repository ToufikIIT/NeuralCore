#include "neuralcore/serialization.h"
#include <stdexcept>

namespace neuralcore {

void save_parameters(const std::string& path,
                     const std::vector<VariablePtr>& params) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for writing: " + path);

    // Write number of parameters
    int num_params = static_cast<int>(params.size());
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(int));

    for (const auto& p : params) {
        // Write number of dimensions
        int ndim = p->data.ndim();
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));

        // Write shape
        for (int d = 0; d < ndim; ++d) {
            int s = p->data.shape()[d];
            file.write(reinterpret_cast<const char*>(&s), sizeof(int));
        }

        // Write data
        Tensor c = p->data.contiguous();
        file.write(reinterpret_cast<const char*>(c.data()),
                   c.size() * sizeof(float));
    }
}

void load_parameters(const std::string& path,
                     std::vector<VariablePtr>& params) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for reading: " + path);

    int num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(int));

    if (num_params != static_cast<int>(params.size())) {
        throw std::runtime_error("Parameter count mismatch");
    }

    for (auto& p : params) {
        int ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(int));

        std::vector<int> shape(ndim);
        for (int d = 0; d < ndim; ++d) {
            file.read(reinterpret_cast<char*>(&shape[d]), sizeof(int));
        }

        int total = 1;
        for (int s : shape) total *= s;

        std::vector<float> data(total);
        file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));

        p->data = Tensor(shape, data);
    }
}

} // namespace neuralcore
