#include "DoubleConv.hpp"

med::layers::DoubleConvImpl::DoubleConvImpl(int inChannels, int outChannels) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 3).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, 3).padding(1)));
}

torch::Tensor med::layers::DoubleConvImpl::forward(torch::Tensor x) {
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    return x;
}
