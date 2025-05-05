#pragma once

#include <torch/torch.h>

namespace med {

namespace layers {

// Double convolution: two 3x3 conv layers with ReLU
class DoubleConvImpl : public torch::nn::Module {
public:
    // Constructor
    DoubleConvImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const DoubleConvImpl& layer) {
        os << "Double convolution: two 3x3 conv layers with ReLU (Used in UNet)";
        return os;
    }
private:
    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
};
TORCH_MODULE(DoubleConv);

} // namespace layers

} // namespace med
