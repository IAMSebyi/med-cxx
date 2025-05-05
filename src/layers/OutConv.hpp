#pragma once

#include <torch/torch.h>

namespace med {

namespace layers {

// Final 1x1 convolution to map to output channels
class OutConvImpl : public torch::nn::Module {
public:
    // Constructor
    OutConvImpl(int inChannels, int outChannels);
    
    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const OutConvImpl& layer) {
        os << "Final 1x1 convolution to map to output channels (Used in UNet)";
        return os;
    }
private:
    // Layers
    torch::nn::Conv2d conv{nullptr};
};
TORCH_MODULE(OutConv);

} // namespace layers

} // namespace med
