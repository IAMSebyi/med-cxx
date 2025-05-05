#pragma once

#include "DoubleConv.hpp"
#include <torch/torch.h>

namespace med {

namespace layers {

// Up-sampling block: ConvTranspose2d then DoubleConv with skip connection
class UpImpl : public torch::nn::Module {
public:
    // Constructor
    UpImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2);

    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const UpImpl& layer) {
        os << "Up-sampling block: ConvTranspose2d then DoubleConv with skip connection (Used in UNet)";
        return os;
    }
    
private:
    // Layers
    torch::nn::ConvTranspose2d up{nullptr};
    DoubleConv conv{nullptr};
};
TORCH_MODULE(Up);

} // namespace layers

} // namespace med
