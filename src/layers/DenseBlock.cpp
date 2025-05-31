#include "DenseBlock.hpp"

namespace med {
namespace layers {

DenseBlockImpl::DenseBlockImpl(int numLayers, int inChannels, int growthRate)
: BaseLayer("DenseBlock: stack of numLayers DenseLayer modules (Used in DenseNet)") {
    int channels = inChannels;
    for (int i = 0; i < numLayers; ++i) {
        DenseLayer layer(channels, growthRate);
        layers->push_back(layer);
        register_module("denselayer_" + std::to_string(i+1), layer);
        channels += growthRate;
    }
    register_module("layers", layers);
}

torch::Tensor DenseBlockImpl::forward(torch::Tensor x) {
    return layers->forward(x);
}

} // namespace layers
} // namespace med
