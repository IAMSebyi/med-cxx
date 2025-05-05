#pragma once

#include "BaseModel.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/DenseBlock.hpp"
#include "layers/Transition.hpp"
#include <torch/torch.h>

namespace med {

namespace models {

// DenseNet implementation class inheriting BaseModel
class DenseNetImpl : public BaseModel {
public:
    // Constructor
    DenseNetImpl(
        const std::vector<int>& blockConfig,
        int growthRate,
        int numInitFeatures = 64,
        int numClasses = 1000,
        torch::Device device = torch::kCPU
    );

    // Forward pass
    torch::Tensor predict(const torch::Tensor& input) override;

private:
    // Layers
    // Initial convolution + pooling
    torch::nn::Conv2d initConv{nullptr};
    torch::nn::BatchNorm2d initBN{nullptr};
    torch::nn::ReLU initReLU;
    torch::nn::MaxPool2d initPool{nullptr};

    // The dense blocks and transitions
    torch::nn::Sequential features{nullptr};

    // Final batch‐norm, pooling, classifier
    torch::nn::BatchNorm2d finalBN{nullptr};
    torch::nn::AdaptiveAvgPool2d avgPool{nullptr};
    torch::nn::Linear classifier{nullptr};

    // Save for use in building the model or forward pass
    std::vector<int> blockConfig;
    int growthRate;
};
TORCH_MODULE(DenseNet);

} // namespace models

} // namespace med
