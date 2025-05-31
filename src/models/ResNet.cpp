#include "ResNet.hpp"

namespace med {
namespace models {

// Explicit instantiation of templates
template class ResNetImpl<med::layers::BasicBlock>;
template class ResNetImpl<med::layers::Bottleneck>;

ResNet::ResNet(Version ver, int numClasses, torch::Device device)
: BaseModel("ResNet", device),
  version_(ver)
{
    if (ver == R18) {
        res18 = std::make_shared<ResNet18Impl>(std::vector<int>{2,2,2,2}, numClasses);
        register_module("resnet", res18);
    }
    else if (ver == R34) {
        res34 = std::make_shared<ResNet34Impl>(std::vector<int>{3,4,6,3}, numClasses);
        register_module("resnet", res34);
    }
    else if (ver == R50) {
        res50 = std::make_shared<ResNet50Impl>(std::vector<int>{3,4,6,3}, numClasses);
        register_module("resnet", res50);
    }
    else if (ver == R101) {
        res101 = std::make_shared<ResNet101Impl>(std::vector<int>{3,4,23,3}, numClasses);
        register_module("resnet", res101);
    }
    else {  // R152
        res152 = std::make_shared<ResNet152Impl>(std::vector<int>{3,8,36,3}, numClasses);
        register_module("resnet", res152);
    }
}

torch::Tensor ResNet::predict(const torch::Tensor& input) {
    switch (version_) {
        case R18: return res18->forward(input);
        case R34: return res34->forward(input);
        case R50: return res50->forward(input);
        case R101: return res101->forward(input);
        case R152: return res152->forward(input);
    }
    // Fallback
    return res18->forward(input);
}

} // namespace models
} // namespace med
