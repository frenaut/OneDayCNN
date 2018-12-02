#ifndef NNBASE_HPP
#define NNBASE_HPP
#include "io.hpp" // Tensor

class NNBase {
public: 
    virtual Tensor Forward(const Tensor & input);
    virtual Tensor Backward(const Tensor & input);
};
#endif