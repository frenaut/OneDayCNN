#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <array>
#include <vector>

#include <Eigen/Dense>

class Tensor {
public:
    Tensor(){};
    Tensor(int width, int height, int depth);
    Tensor(Eigen::MatrixXf matrix);
    Tensor(const std::array<int, 3> & input_size);
    std::vector<Eigen::MatrixXf> data() const;
    std::array<int, 3> size() const; 
    void setData(Eigen::MatrixXf matrix, int depth);

private:
    std::vector<Eigen::MatrixXf> data_;
    int width_;
    int height_;
    int depth_;
};
// overload stream operator
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

#endif

