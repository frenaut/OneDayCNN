#ifndef NNBASE_HPP
#define NNBASE_HPP
#include "io.hpp" // Tensor

class NNBase {
public:
    virtual Tensor forward(const Tensor & input){};
    virtual Tensor backward(const Tensor & input){};
};
#endif