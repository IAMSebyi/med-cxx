#pragma once

#include "BaseModel.hpp"
#include <torch/torch.h>

// Single dense layer: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv, then concatenate input & output
struct DenseLayerImpl : torch::nn::Module {
    // Layers
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    int growthRate;

    // Constructor
    DenseLayerImpl(int inChannels, int growthRate)
      : growthRate(growthRate) {
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(inChannels));
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, growthRate * 4, 1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(growthRate * 4));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(growthRate * 4, growthRate, 3).padding(1).bias(false)));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        auto out = conv1->forward(torch::relu(bn1->forward(x)));
        out = conv2->forward(torch::relu(bn2->forward(out)));
        return torch::cat({x, out}, 1);
    }
};
TORCH_MODULE(DenseLayer);

// DenseBlock: stack of numLayers DenseLayer modules
struct DenseBlockImpl : torch::nn::Module {
    // Layers
    torch::nn::Sequential layers;

    // Constructor
    DenseBlockImpl(int numLayers, int inChannels, int growthRate) {
        int channels = inChannels;
        for (int i = 0; i < numLayers; ++i) {
            DenseLayer layer(channels, growthRate);
            layers->push_back(layer);
            register_module("denselayer_" + std::to_string(i+1), layer);
            channels += growthRate;
        }
        register_module("layers", layers);
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};
TORCH_MODULE(DenseBlock);

// Transition: BN -> ReLU -> 1x1 Conv -> 2x2 Avg Pool
struct TransitionImpl : torch::nn::Module {
    // Layers
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::AvgPool2d pool{nullptr};

    // Constructor
    TransitionImpl(int inChannels, int outChannels) {
        bn = register_module("bn",   torch::nn::BatchNorm2d(inChannels));
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1).bias(false)));
        pool = register_module("pool", torch::nn::AvgPool2d(2));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(torch::relu(bn->forward(x)));
        return pool->forward(x);
    }
};
TORCH_MODULE(Transition);

// DenseNet class inheriting BaseModel
class DenseNet : public BaseModel {
public:
    // Constructor
    DenseNet(
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
