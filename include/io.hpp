#include <Eigen/Dense>
#include <string>
#include <array>
#include <iostream>

template <int depth>
class Tensor {
public:
    Tensor(int width, int height) {
        for (int d = 0; d < depth; ++d) {
            data[d] = Eigen::MatrixXf(height, width);
        }
    }
    
    std::array<Eigen::MatrixXf, depth> data;  
};

namespace io {
    template <int depth>
    Tensor<depth> matrixFromPng(const std::string & filename, int width, int height);
}