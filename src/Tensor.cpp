#include "../include/Tensor.hpp"

Tensor::Tensor(int width, int height, int depth) : width_(width), height_(height), depth_(depth) {
  assert(depth_ <= 3);
  for (int d = 0; d < depth; ++d) {
      data_[d] = Eigen::MatrixXf(height, width);
  }
} 

Tensor::Tensor(const std::array<int, 3> & input_size) : Tensor(input_size[0], input_size[1], input_size[2]){}


std::array<Eigen::MatrixXf, 3> Tensor::data() const {
  return data_; 
}

std::array<int, 3> Tensor::size() const {
  return std::array<int, 3> {{width_, height_, depth_}};
}

void Tensor::setData(const Eigen::MatrixXf & matrix, int depth){
  assert(depth_ <= 3);
  data_[depth] = matrix;
}