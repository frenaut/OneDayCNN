#include "../include/Tensor.hpp"
#include "Eigen/Dense"
#include "../include/NNFC.hpp"
#include <iostream>


NNFC::NNFC(int output_depth) : output_depth_(output_depth) {
  initialized_ = false;
  weights_.setDepth(output_depth_);
  biases_.setDepth(output_depth_);
}

Tensor NNFC::forward(const Tensor & input) {
  /*
  Forward pass for fully-convolutional layer
  if not initialized, it currently sets the weights and biases randomly
  once initialized, the output for depth z is the sum of 
   */
  std::cout << "forward pass for FC layer" << std::endl;
  std::array<int, 3> input_size = input.size();

  if (!initialized_) {
    // initialize weights and biases tensor to match input tensor size and output depth
    Eigen::MatrixXf input_mat(input_size[1], input_size[0]); // height, then width
    for (int z = 0; z < output_depth_; ++z) {
      weights_.setData(input_mat.setRandom(), z);
      Eigen::MatrixXf simple_float(1, 1); 
      biases_.setData(simple_float.setRandom(), z);
    }
    initialized_ = true;
  }

  Tensor output(1, 1, output_depth_);
  auto input_data = input.data();
  auto weights_data = weights_.data();

  for (int out_z=0; out_z < output_depth_; ++out_z) {
    Eigen::MatrixXf output_at_z(1, 1);
    output_at_z << 0;
    for(int in_z = 0; in_z < input.size()[2]; ++in_z) {
      output_at_z << output_at_z(0, 0) + input_data[in_z].cwiseProduct(weights_data[out_z]).sum();
    }
    output_at_z += biases_.data()[out_z];
    output.setData(output_at_z, out_z);
    
  }
  return output;
}
