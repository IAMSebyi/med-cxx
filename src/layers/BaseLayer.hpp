#pragma once

#include <torch/torch.h>

namespace med {
namespace layers {

// Base layer class for all layers
class BaseLayer : public torch::nn::Module {
public:
    // Constructor
    BaseLayer(const std::string& name);

    // Destructor
    virtual ~BaseLayer() = default;

    // Forward pass (pure virtual function)
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const BaseLayer& layer) {
        os << "Layer: " << layer.name;
        return os;
    }

private:
    std::string name; // Name of the layer
};

} // namespace layers
} // namespace med
