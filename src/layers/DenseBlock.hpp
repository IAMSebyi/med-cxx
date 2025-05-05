#pragma once

#include "DenseLayer.hpp"
#include <torch/torch.h>

namespace med {

namespace layers {

// DenseBlock: stack of numLayers DenseLayer modules
class DenseBlockImpl : public torch::nn::Module {
public:
    // Constructor
    DenseBlockImpl(int numLayers, int inChannels, int growthRate);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const DenseBlockImpl& layer) {
        os << "DenseBlock: stack of numLayers DenseLayer modules (Used in DenseNet)";
        return os;
    }
private:
    // Layers
    torch::nn::Sequential layers;
};
TORCH_MODULE(DenseBlock);

} // namespace layers

} // namespace med
