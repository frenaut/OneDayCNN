#ifndef NNInput_HPP
#define NNInput_HPP
#include <string>
#include <vector>
#include <tuple>
#include "NNBase.hpp"

class NNInput : public NNBase {
public:
    NNInput(std::string mnist_images, std::string mnist_labels) 
    : mnist_images_(mnist_images), mnist_labels_(mnist_labels) {};

    Tensor forward();
    
    Tensor forward(const Tensor & input);

    Tensor backward(const Tensor & input);

    void loadMNISTData();

private:
    std::string mnist_images_;
    std::string mnist_labels_;
    int number_of_images_;
    int number_of_labels_;
    std::vector<std::tuple<Eigen::MatrixXf, int>>samples_;
    int iterator_ = 0;
};
#endif