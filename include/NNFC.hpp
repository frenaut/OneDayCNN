#ifndef NNFC_HPP
#define NNFC_HPP

#include "Tensor.hpp"
#include "NNBase.hpp"

class NNFC : public NNBase {
  public:
   // constructor taking hyperparameters
   NNFC(int output_depth);
   // forward pass
   Tensor forward(const Tensor & input);
   // backward pass
   Tensor backward(const Tensor & input){};

  private:
    // weights
    Tensor weights_;
    Tensor biases_;
    
    // hyperparams
    int output_depth_;
    bool initialized_;
};

#endif // NNFC_HPP