#include "DenseNet.hpp"

DenseNet::DenseNet(
    const std::vector<int>& blockConfig_,
    int growthRate_,
    int numInitFeatures,
    int numClasses,
    torch::Device device
) : BaseModel("DenseNet", device),
    blockConfig(blockConfig_),
    growthRate(growthRate_),
    initReLU(torch::nn::ReLUOptions(true)),
    initPool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
    avgPool(torch::nn::AdaptiveAvgPool2dOptions({1,1}))
{
    // Initial conv + BN
    initConv = register_module("init_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, numInitFeatures, 7).stride(2).padding(3).bias(false)));
    initBN = register_module("init_bn", torch::nn::BatchNorm2d(numInitFeatures));

    // Build features
    features = register_module("features", torch::nn::Sequential());
    int numFeatures = numInitFeatures;

    for (size_t i = 0; i < blockConfig.size(); ++i) {
        // DenseBlock
        DenseBlock block(blockConfig[i], numFeatures, growthRate);
        features->push_back(block);
        numFeatures += blockConfig[i] * growthRate;

        // Transition (except after last)
        if (i + 1 < blockConfig.size()) {
            int outFeatures = numFeatures / 2;
            Transition trans(numFeatures, outFeatures);
            features->push_back(trans);
            numFeatures = outFeatures;
        }
    }

    // Final BN + classifier
    finalBN = register_module("final_bn", torch::nn::BatchNorm2d(numFeatures));
    classifier = register_module("classifier", torch::nn::Linear(numFeatures, numClasses));
}

torch::Tensor DenseNet::predict(const torch::Tensor& input) {
    // Initial layers
    auto out = initPool->forward(initReLU->forward(initBN->forward(initConv->forward(input))));
    // Forward through the sequential of blocks/transitions
    out = features->forward(out);
    // Final batchnorm, pooling, flatten + linear
    out = torch::relu(finalBN->forward(out));
    out = avgPool->forward(out);
    out = out.view({out.size(0), -1});
    return classifier->forward(out);
}
