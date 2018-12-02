#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <array>

#include <Eigen/Dense>

class Tensor {
public:
    Tensor(int width, int height, int depth);
    std::array<Eigen::MatrixXf, 3> data;  
};
#endif