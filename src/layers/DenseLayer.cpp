#include "DenseLayer.hpp"

namespace med {
namespace layers {

// Constructor
med::layers::DenseLayerImpl::DenseLayerImpl(int inChannels, int growthRate)
: BaseLayer("Single dense layer: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv, then concatenate input & output (Used in DenseNet)"), growthRate(growthRate) {
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(inChannels));
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, growthRate * 4, 1).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(growthRate * 4));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(growthRate * 4, growthRate, 3).padding(1).bias(false)));
}

// Forward pass
torch::Tensor med::layers::DenseLayerImpl::forward(torch::Tensor x) {
    auto out = conv1->forward(torch::relu(bn1->forward(x)));
    out = conv2->forward(torch::relu(bn2->forward(out)));
    return torch::cat({x, out}, 1);
}

} // namespace layers
} // namespace med
