#include "../include/Tensor.hpp"

Tensor::Tensor(int width, int height, int depth) : width_(width), height_(height), depth_(depth) {
  data_.resize(depth);
  for (int d = 0; d < depth; ++d) {
      data_[d] = Eigen::MatrixXf(height, width);
  }
} 

Tensor::Tensor(const std::array<int, 3> & input_size) : Tensor(input_size[0], input_size[1], input_size[2]){}

Tensor::Tensor(Eigen::MatrixXf matrix) {
    width_ = matrix.cols();
    height_ = matrix.rows();
    data_[0] = matrix;
};

std::vector<Eigen::MatrixXf> Tensor::data() const {
  return data_; 
}

std::array<int, 3> Tensor::size() const {
  return std::array<int, 3> {{width_, height_, depth_}};
}

void Tensor::setData(Eigen::MatrixXf matrix, int depth){
  assert(depth_ < data_.size());
  data_[depth] = matrix;
}