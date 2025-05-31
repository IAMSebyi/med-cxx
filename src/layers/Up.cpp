#include "Up.hpp"

namespace med {
namespace layers {

med::layers::UpImpl::UpImpl(int inChannels, int outChannels) 
: BaseLayer("Up-sampling block: ConvTranspose2d then DoubleConv with skip connection (Used in UNet)") {
    up = register_module("up", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(inChannels, inChannels / 2, 2).stride(2)));
    conv = register_module("conv", DoubleConv(inChannels, outChannels));
}

torch::Tensor med::layers::UpImpl::forward_(torch::Tensor x1, torch::Tensor x2) {
    x1 = up->forward(x1);
    auto diffY = x2.size(2) - x1.size(2);
    auto diffX = x2.size(3) - x1.size(3);
    x1 = torch::constant_pad_nd(x1, {diffX / 2, diffX - diffX / 2, diffY /2 , diffY - diffY / 2});
    auto x = torch::cat({x2, x1}, 1);
    return conv->forward(x);
}

} // namespace layers
} // namespace med
