#include "Down.hpp"

med::layers::DownImpl::DownImpl(int inChannels, int outChannels) {
    pool = register_module("pool", torch::nn::MaxPool2d(2));
    conv = register_module("conv", DoubleConv(inChannels, outChannels));
}

torch::Tensor med::layers::DownImpl::forward(torch::Tensor x) {
    x = pool->forward(x);
    return conv->forward(x);
}