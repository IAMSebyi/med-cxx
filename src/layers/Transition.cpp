#include "Transition.hpp"

namespace med {
namespace layers {

TransitionImpl::TransitionImpl(int inChannels, int outChannels) 
: BaseLayer("Transition: BN -> ReLU -> 1x1 Conv -> 2x2 Avg Pool (Used in DenseNet)") {
    bn = register_module("bn", torch::nn::BatchNorm2d(inChannels));
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1).bias(false)));
    pool = register_module("pool", torch::nn::AvgPool2d(2));
}

torch::Tensor TransitionImpl::forward(torch::Tensor x) {
    x = conv->forward(torch::relu(bn->forward(x)));
    return pool->forward(x);
}

} // namespace layers
} // namespace med
