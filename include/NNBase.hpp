#ifndef NNBASE_HPP
#define NNBASE_HPP
#include "io.hpp" // Tensor

class NNBase {
public:
    virtual Tensor forward(const Tensor & input) = 0;
    virtual Tensor backward(const Tensor & input) = 0;
};
#endif