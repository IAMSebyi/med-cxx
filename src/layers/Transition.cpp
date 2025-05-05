#include "Transition.hpp"

med::layers::TransitionImpl::TransitionImpl(int inChannels, int outChannels) {
    bn = register_module("bn", torch::nn::BatchNorm2d(inChannels));
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1).bias(false)));
    pool = register_module("pool", torch::nn::AvgPool2d(2));
}

torch::Tensor med::layers::TransitionImpl::forward(torch::Tensor x) {
    x = conv->forward(torch::relu(bn->forward(x)));
    return pool->forward(x);
}