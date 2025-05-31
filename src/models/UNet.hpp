#pragma once

#include "BaseModel.hpp"
#include "layers/DoubleConv.hpp"
#include "layers/Down.hpp"
#include "layers/Up.hpp"
#include "layers/OutConv.hpp"
#include <torch/torch.h>

namespace med {
    
namespace models {

// UNet implementation class inheriting BaseModel
class UNetImpl : public BaseModel {
public:
    // Constructor
    UNetImpl(int inChannels, int outChannels, torch::Device device = torch::kCPU);

    // Forward pass
    torch::Tensor predict(const torch::Tensor& input) override;

private:
    // Layers
    med::layers::DoubleConv inc;
    med::layers::Down down1, down2, down3, down4;
    med::layers::Up up1, up2, up3, up4;
    med::layers::OutConv outc;
};
TORCH_MODULE(UNet);

} // namespace models

} // namespace med
