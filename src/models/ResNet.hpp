#pragma once

#include "BaseModel.hpp"
#include "layers/BasicBlock.hpp"
#include "layers/Bottleneck.hpp"
#include <torch/torch.h>

namespace med {

namespace models {

// Templated ResNet body
template <class Block>
class ResNetImpl : public torch::nn::Module {
public:
    // Constructor
    ResNetImpl(const std::vector<int>& layers, int numClasses=1000)
    : relu(torch::nn::ReLUOptions(true)),
      maxpool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
      avgpool(torch::nn::AdaptiveAvgPool2dOptions({1,1})) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        inplanes = 64;

        layer1 = register_module("layer1", _make_layer(64,  layers[0], 1));
        layer2 = register_module("layer2", _make_layer(128, layers[1], 2));
        layer3 = register_module("layer3", _make_layer(256, layers[2], 2));
        layer4 = register_module("layer4", _make_layer(512, layers[3], 2));

        fc = register_module("fc", torch::nn::Linear(512 * Block::expansion, numClasses));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = maxpool->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avgpool->forward(x).view({x.size(0), -1});
        return fc->forward(x);
    }

private:
    int inplanes;
    // Layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::ReLU relu;
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};

    torch::nn::Sequential _make_layer(int planes, int blocks, int stride) {
        torch::nn::Sequential downsample;
        if (stride != 1 || inplanes != planes * Block::expansion) {
            downsample = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * Block::expansion, 1).stride(stride).bias(false)),
            torch::nn::BatchNorm2d(planes * Block::expansion)
            );
        }
        torch::nn::Sequential seq;
        seq->push_back(Block(inplanes, planes, stride, downsample));
        inplanes = planes * Block::expansion;
        for (int i = 1; i < blocks; ++i) {
            seq->push_back(Block(inplanes, planes));
        }
        return seq;
    }
};

// Shared‐ptr aliases
using ResNet18Impl = ResNetImpl<med::layers::BasicBlock>;
using ResNet34Impl = ResNetImpl<med::layers::BasicBlock>;
using ResNet50Impl = ResNetImpl<med::layers::Bottleneck>;
using ResNet101Impl = ResNetImpl<med::layers::Bottleneck>;
using ResNet152Impl = ResNetImpl<med::layers::Bottleneck>;

// Public ResNet
class ResNet : public BaseModel {
public:
    enum Version { R18, R34, R50, R101, R152 };

    // Constructor
    ResNet(Version ver, int numClasses = 1000, torch::Device device = torch::kCPU);

    // Forward pass
    torch::Tensor predict(const torch::Tensor& input) override;

private:
    Version version_;
    std::shared_ptr<ResNet18Impl> res18;
    std::shared_ptr<ResNet34Impl> res34;
    std::shared_ptr<ResNet50Impl> res50;
    std::shared_ptr<ResNet101Impl> res101;
    std::shared_ptr<ResNet152Impl> res152;
};

} // namespace models

} // namespace med
