#ifndef IO_HPP
#define IO_HPP
#include <string>

#include "Tensor.hpp"

namespace io {
    Tensor matrixFromPng(const std::string & filename, int width, int height, int depth);
}
#endif