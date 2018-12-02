#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <array>

#include <Eigen/Dense>

class Tensor {
public:
    Tensor(int width, int height, int depth);
    Tensor(const std::array<int, 3> & input_size);
    std::array<Eigen::MatrixXf, 3> data() const;
    std::array<int, 3> size() const; 
    void setData(const Eigen::MatrixXf & matrix, int depth);

private:
    std::array<Eigen::MatrixXf, 3> data_;
    int width_;
    int height_;
    int depth_;
};
#endif