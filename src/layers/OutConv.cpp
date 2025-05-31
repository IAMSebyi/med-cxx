#include "OutConv.hpp"

namespace med {
namespace layers {

OutConvImpl::OutConvImpl(int inChannels, int outChannels) 
: BaseLayer("Final 1x1 convolution to map to output channels (Used in UNet)") {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1)));
}

torch::Tensor OutConvImpl::forward(torch::Tensor x) {
    return conv->forward(x);
}

} // namespace layers
} // namespace med
