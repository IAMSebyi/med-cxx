#include "Down.hpp"

namespace med {
namespace layers {

med::layers::DownImpl::DownImpl(int inChannels, int outChannels) 
: BaseLayer("Down-sampling block: MaxPool2d then DoubleConv (Used in UNet)") {
    pool = register_module("pool", torch::nn::MaxPool2d(2));
    conv = register_module("conv", DoubleConv(inChannels, outChannels));
}

torch::Tensor med::layers::DownImpl::forward(torch::Tensor x) {
    x = pool->forward(x);
    return conv->forward(x);
}

} // namespace layers
} // namespace med
