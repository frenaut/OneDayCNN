#include "../include/Tensor.hpp"

Tensor::Tensor(int width, int height, int depth) {
  assert(depth <= 3);
  for (int d = 0; d < depth; ++d) {
      data[d] = Eigen::MatrixXf(height, width);
  }
} 