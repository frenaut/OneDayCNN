#include <vector>
#include <iostream>

#include "../include/NNetwork.hpp"
#include "../include/NNBase.hpp"

NNetwork::NNetwork(std::vector<std::shared_ptr<NNBase>> network) : network_(network) {}

Tensor NNetwork::forward(){
  Tensor state;
  for(auto & layer : network_) {
    auto newState = layer->forward(state);
    std::cout << "Ran through layer " << std::endl;
    state = newState;
  }
  return state;
}