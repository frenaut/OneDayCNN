#ifndef NNetwork_HPP
#define NNetwork_HPP

#include "vector"

#include "Tensor.hpp"
#include "NNBase.hpp"

class NNetwork{
  public:
   // constructor taking vector of NNBase layers
   NNetwork(std::vector<NNBase> network);
   // forward pass
   Tensor forward();
   // backward pass
   Tensor backward(){};

  private:
    std::vector<NNBase> network_;
};

#endif // NNetwork_HPP