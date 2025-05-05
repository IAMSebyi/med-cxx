#pragma once

#include "DoubleConv.hpp"
#include <torch/torch.h>

namespace med {

namespace layers {

// Down-sampling block: MaxPool2d then DoubleConv
class DownImpl : public torch::nn::Module {
public:
    // Constructor
    DownImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const DownImpl& layer) {
        os << "Down-sampling block: MaxPool2d then DoubleConv (Used in UNet)";
        return os;
    }
private:
    // Layers
    torch::nn::MaxPool2d pool{nullptr};
    DoubleConv conv{nullptr};
};
TORCH_MODULE(Down);

} // namespace layers

} // namespace med
