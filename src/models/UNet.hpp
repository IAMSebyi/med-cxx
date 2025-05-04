#pragma once

#include "BaseModel.hpp"
#include <torch/torch.h>

// Double convolution: two 3x3 conv layers with ReLU
struct DoubleConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};

    // Constructor
    DoubleConvImpl(int inChannels, int outChannels) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, 3).padding(1)));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        return x;
    }
};
TORCH_MODULE(DoubleConv);

// Down-sampling block: MaxPool2d then DoubleConv
struct DownImpl : torch::nn::Module {
    torch::nn::MaxPool2d pool{nullptr};
    DoubleConv conv{nullptr};

    // Constructor
    DownImpl(int inChannels, int outChannels) {
        pool = register_module("pool", torch::nn::MaxPool2d(2));
        conv = register_module("conv", DoubleConv(inChannels, outChannels));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = pool->forward(x);
        return conv->forward(x);
    }
};
TORCH_MODULE(Down);

// Up-sampling block: ConvTranspose2d then DoubleConv with skip connection
struct UpImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d up{nullptr};
    DoubleConv conv{nullptr};

    // Constructor
    UpImpl(int inChannels, int outChannels) {
        up = register_module("up", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(inChannels, inChannels / 2, 2).stride(2)));
        conv = register_module("conv", DoubleConv(inChannels, outChannels));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2) {
        x1 = up->forward(x1);
        auto diffY = x2.size(2) - x1.size(2);
        auto diffX = x2.size(3) - x1.size(3);
        x1 = torch::constant_pad_nd(x1, {diffX / 2, diffX - diffX / 2, diffY /2 , diffY - diffY / 2});
        auto x = torch::cat({x2, x1}, 1);
        return conv->forward(x);
    }
};
TORCH_MODULE(Up);

// Final 1x1 convolution to map to output channels
struct OutConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};

    // Constructor
    OutConvImpl(int inChannels, int outChannels) {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1)));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        return conv->forward(x);
    }
};
TORCH_MODULE(OutConv);

// UNet class inheriting BaseModel and torch::nn::Module
class UNet : public BaseModel {
public:
    // Constructor
    UNet(int inChannels, int outChannels, torch::Device device = torch::kCPU);

    // Forward pass
    torch::Tensor predict(const torch::Tensor& input) override;

private:
    // Layers
    DoubleConv inc;
    Down down1, down2, down3, down4;
    Up up1, up2, up3, up4;
    OutConv outc;
};
