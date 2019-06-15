/*
 * A simple libpng example program
 * http://zarb.org/~gc/html/libpng.html
 *
 * Modified by Yoshimasa Niwa to make it much simpler
 * and support all defined color_type.
 *
 * To build, use the next instruction on OS X.
 * $ brew install libpng
 * $ clang -lz -lpng15 libpng_test.c
 *
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <Eigen/Dense>
#include <string>
#include <iostream>

#include "../include/io.hpp"
#include "../include/lodepng.h"
#include "../include/Tensor.hpp"
#include "../include/NNInput.hpp"
#include "../include/NNFC.hpp"
#include "../include/NNetwork.hpp"


int io::writeImage(const std::string & filename, const Tensor & data) {
  std::vector<unsigned char> image;
  const int width = data.size()[0];
  const int height = data.size()[1];
  const int depth = data.size()[2];
  image.resize(width * height * depth);

  // Write tensor data to image buffer
  for (unsigned z = 0; z < depth; ++z) {
    for (unsigned y = 0; y < height; y++) {
      for (unsigned x = 0; x < width; x++) {
        const unsigned int pixel_idx = depth * height * y + depth * x; 
        image[pixel_idx + z] = data.data()[z](y, x);
      }
    }    
  }
  
  // Write tensor data to file
  unsigned error = lodepng::encode(filename, image, width, height, (depth == 1) ? LodePNGColorType::LCT_GREY : LodePNGColorType::LCT_RGB);
  if (error) std::cout << "Write image failed with encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  return static_cast<int>(error);
};

Tensor io::readImage(const std::string & filename, const unsigned int depth) {
  assert(depth == 1 || depth == 3);

  // Read image
  std::vector<unsigned char> image;
  unsigned width, height;
  unsigned error = lodepng::decode(image, width, height, filename, (depth == 1) ? LodePNGColorType::LCT_GREY : LodePNGColorType::LCT_RGB);
  if (error) std::cout << "Read image failed with decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

  // Convert image to tensor
  Tensor data(width, height, depth);
  for (unsigned z = 0; z < depth; ++z) {
    Eigen::MatrixXf matrix(height, width);
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        const unsigned int pixel_idx = depth * width * y + depth * x; 
        matrix(y, x) = image[pixel_idx + z];  
      }
    }
    data.setData(matrix, z);
  }
  return data;
};

int main(int argc, char *argv[]) {

  std::shared_ptr<NNInput> input(new NNInput("/Users/Brinck/Work/OneDayCNN/train-images-idx3-ubyte", "/Users/Brinck/Work/OneDayCNN/train-labels-idx1-ubyte"));
  input->loadMNISTData();
  
  std::shared_ptr<NNFC> output(new NNFC(50));
  std::shared_ptr<NNFC> output_2(new NNFC(10));
  std::cout << " output defined" << std::endl;
  std::vector<std::shared_ptr<NNBase>> layers = {input, output, output_2};
  NNetwork network(layers);
  std::cout << " network defined" << std::endl;
  std::cout << network.forward() << std::endl;

  return 0;
}