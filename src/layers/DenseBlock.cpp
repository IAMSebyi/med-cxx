#include "DenseBlock.hpp"

med::layers::DenseBlockImpl::DenseBlockImpl(int numLayers, int inChannels, int growthRate) {
    int channels = inChannels;
    for (int i = 0; i < numLayers; ++i) {
        DenseLayer layer(channels, growthRate);
        layers->push_back(layer);
        register_module("denselayer_" + std::to_string(i+1), layer);
        channels += growthRate;
    }
    register_module("layers", layers);
}

torch::Tensor med::layers::DenseBlockImpl::forward(torch::Tensor x) {
    return layers->forward(x);
}
