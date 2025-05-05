#include <torch/torch.h>

namespace med {

namespace layers {

// Transition: BN -> ReLU -> 1x1 Conv -> 2x2 Avg Pool
class TransitionImpl : public torch::nn::Module {
public:
    // Constructor
    TransitionImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const TransitionImpl& layer) {
        os << "Transition: BN -> ReLU -> 1x1 Conv -> 2x2 Avg Pool (Used in DenseNet)";
        return os;
    }
    
private:
    // Layers
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::AvgPool2d pool{nullptr};
};
TORCH_MODULE(Transition);

} // namespace layers

} // namespace med
