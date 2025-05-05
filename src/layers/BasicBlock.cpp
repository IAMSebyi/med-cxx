#include "BasicBlock.hpp"

med::layers::BasicBlock::BasicBlock(int inPlanes, int planes, int stride, torch::nn::Sequential downsample_)
    : downsample(downsample_) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inPlanes, planes, 3).stride(stride).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));
    if (!downsample_->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor med::layers::BasicBlock::forward(torch::Tensor x) {
    auto identity = x.clone();
    x = torch::relu(bn1->forward(conv1->forward(x)));
    x = torch::relu(bn2->forward(conv2->forward(x)));
    if (!downsample->is_empty()) {
        identity = downsample->forward(identity);
    }
    x += identity;
    return torch::relu(x);
}
