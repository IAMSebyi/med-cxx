#pragma once

#include "BaseLayer.hpp"
#include "DenseLayer.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// DenseBlock: stack of numLayers DenseLayer modules
class DenseBlockImpl : public BaseLayer {
public:
    // Constructor
    DenseBlockImpl(int numLayers, int inChannels, int growthRate);

    // Forward pass
    torch::Tensor forward(torch::Tensor x) override;

private:
    // Layers
    torch::nn::Sequential layers;
};
TORCH_MODULE(DenseBlock);

} // namespace layers
} // namespace med
