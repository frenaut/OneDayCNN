#ifndef IO_HPP
#define IO_HPP
#include <string>

#include "Tensor.hpp"

namespace io {
    Tensor matrixFromPng(const std::string & filename, int width, int height, int depth);

    int writeImage(const std::string & filename, const Tensor & data);

    Tensor readImage(const std::string & filename, const unsigned int depth = 1);
}
#endif