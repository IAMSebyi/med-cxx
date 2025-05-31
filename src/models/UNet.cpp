#include "UNet.hpp"

namespace med {
namespace models {

UNetImpl::UNetImpl(int inChannels, int outChannels, torch::Device device)
    : BaseModel("UNet", device),
      inc(inChannels, 64),
      down1(64, 128),
      down2(128, 256),
      down3(256, 512),
      down4(512, 1024),
      up1(1024, 512),
      up2(512, 256),
      up3(256, 128),
      up4(128, 64),
      outc(64, outChannels)
{
    // Register all submodules
    register_module("inc", inc);
    register_module("down1", down1);
    register_module("down2", down2);
    register_module("down3", down3);
    register_module("down4", down4);
    register_module("up1", up1);
    register_module("up2", up2);
    register_module("up3", up3);
    register_module("up4", up4);
    register_module("outc", outc);
}

torch::Tensor UNetImpl::predict(const torch::Tensor& input) {
    auto x1 = inc->forward(input);
    auto x2 = down1->forward(x1);
    auto x3 = down2->forward(x2);
    auto x4 = down3->forward(x3);
    auto x5 = down4->forward(x4);
    auto y1 = up1->forward_(x5, x4);
    auto y2 = up2->forward_(y1, x3);
    auto y3 = up3->forward_(y2, x2);
    auto y4 = up4->forward_(y3, x1);
    return outc->forward(y4);
}

} // namespace models
} // namespace med
