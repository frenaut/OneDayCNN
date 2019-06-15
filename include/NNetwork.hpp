#ifndef NNetwork_HPP
#define NNetwork_HPP

#include "vector"
#include <memory>

#include "Tensor.hpp"
#include "NNBase.hpp"

class NNetwork{
  public:
   // constructor taking vector of NNBase layers
   NNetwork(std::vector<std::shared_ptr<NNBase>> network);

   // forward pass
   Tensor forward();
   // backward pass
   Tensor backward(){};

  private:
    std::vector<std::shared_ptr<NNBase>> network_;
};

#endif // NNetwork_HPP