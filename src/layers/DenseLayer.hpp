#pragma once

#include <torch/torch.h>

namespace med {

namespace layers {

// Single dense layer: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv, then concatenate input & output
class DenseLayerImpl : public torch::nn::Module {
public:
    // Constructor
    DenseLayerImpl(int inChannels, int growthRate);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const DenseLayerImpl& layer) {
        os << "Single dense layer: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv, then concatenate input & output (Used in DenseNet)";
        return os;
    }
private:
    // Layers
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    int growthRate;
};
TORCH_MODULE(DenseLayer);

} // namespace layers

} // namespace med