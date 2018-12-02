#include "../include/Tensor.hpp"
#include "Eigen/Dense"
#include "../include/NNFC.hpp"


NNFC::NNFC(int output_depth) : output_depth_(output_depth) {
  initialized_ = false;
}

Tensor NNFC::forward(const Tensor & input) {
  std::array<int, 3> input_size = input.size();

  if (!initialized_) {
    // initialize weights and biases tensor to match input tensor size and output depth
    Eigen::MatrixXf input_mat(input_size[1], input_size[0]); // height, then width
    for (int z = 0; z < output_depth_; ++z) {
      weights_.setData(input_mat.setRandom(), output_depth_);
      Eigen::MatrixXf simple_float(1, 1); 
      biases_.setData(simple_float.setRandom(), output_depth_);
    }
    initialized_ = true;
  }

  Tensor output(1, 1, output_depth_);
  auto input_data = input.data();
  auto weights_data = weights_.data();
  for(int z = 0; z < input.size()[2]; ++z) {
    Eigen::MatrixXf output_at_z(1, 1);
    output_at_z << input_data[z].cwiseProduct(weights_data[z]).sum();
    output_at_z += biases_.data()[z];
    output.setData(output_at_z, z);
  }

  return output;
}
