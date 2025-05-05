#include "OutConv.hpp"

med::layers::OutConvImpl::OutConvImpl(int inChannels, int outChannels) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1)));
}

torch::Tensor med::layers::OutConvImpl::forward(torch::Tensor x) {
    return conv->forward(x);
}
