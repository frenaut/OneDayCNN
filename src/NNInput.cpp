#include <string>
#include <fstream>
#include <tuple>
#include <stdexcept>
#include "NNInput.hpp"
#include <iostream>
Tensor NNInput::forward() {
    if (!samples_.size()) throw std::runtime_error("You must load a data set before calling forward on the Input layer");
    if (iterator_ >= samples_.size()) iterator_ = 0;

    auto image = std::get<0>(samples_[iterator_]);
    return Tensor(image);
}

Tensor NNInput::forward(const Tensor & input) {
    return forward();
}

Tensor NNInput::backward(const Tensor & input) {
    return Tensor(0, 0, 0);
}

void NNInput::loadMNISTData() {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;
    std::ifstream image_file(mnist_images_, std::ios::binary);
    std::ifstream labels_file(mnist_labels_, std::ios::binary);
    std::cout << image_file.is_open() << " " << labels_file.is_open() << std::endl;
    if(!image_file.is_open()) throw std::runtime_error("Cannot open file `" + mnist_images_ + "`!");
    if(!labels_file.is_open()) throw std::runtime_error("Cannot open file `" + mnist_labels_ + "`!");

    int magic_number = 0, n_rows = 0, n_cols = 0;

    image_file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");
    labels_file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

    labels_file.read((char *)&number_of_labels_, sizeof(number_of_labels_)), number_of_labels_ = reverseInt(number_of_labels_);
    image_file.read((char *)&number_of_images_, sizeof(number_of_images_)), number_of_images_ = reverseInt(number_of_images_);
    assert(number_of_images_ == number_of_labels_);

    image_file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    image_file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
    int image_size = n_rows * n_cols;
    for(int i = 0; i < number_of_images_; i++) {
        uchar* image = new uchar[image_size];
        image_file.read((char *)image, image_size);
        uchar label = 0;
        labels_file.read((char*)& label, 1);
        Eigen::MatrixXf mnist_image(n_rows, n_cols);
        for (int y = 0; y < n_rows; ++y) {
            for (int x = 0; x < n_cols; ++x) {
                mnist_image(y, x) = static_cast<float>(image[y * x]);
            }
        }
        samples_.emplace_back(mnist_image, label);
    }
}
