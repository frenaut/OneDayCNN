#include "vector"
#include "../include/NNetwork.hpp"
#include "../include/NNBase.hpp"

NNetwork::NNetwork(std::vector<NNBase> network) : network_(network) {}

Tensor NNetwork::forward(){
  Tensor state;
  for(auto & layer: network_) {
    state = layer.forward(state);
  }
  return state;
}