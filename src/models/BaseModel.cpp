#include "BaseModel.hpp"

BaseModel::BaseModel(const std::string& modelName, int inChannels, int outChannels)
    : name(modelName), device(torch::kCPU) // default device set to CPU
{
    net = torch::nn::Sequential(
        // First conv layer: from inChannels to 16 feature maps, kernel size 3, stride 1, padding 1
        torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, 16, 3).stride(1).padding(1)),
        torch::nn::ReLU(),
        // Second conv layer: from 16 to outChannels feature maps
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, outChannels, 3).stride(1).padding(1))
    );
    net->to(device);
}

BaseModel::BaseModel(const BaseModel& other)
    : name(other.name), device(other.device)
{
    net = torch::nn::Sequential();
    for (const auto& module : *other.net) {
        net->push_back(module.clone());
    }
}

BaseModel& BaseModel::operator=(const BaseModel& other) {
    if (this != &other) {
        name = other.name;
        device = other.device;
        net = torch::nn::Sequential();
        for (const auto& module : *other.net) {
            net->push_back(module.clone());
        }
    }
    return *this;
}

BaseModel::~BaseModel() {}

void BaseModel::trainModel(const torch::Tensor& inputs, const torch::Tensor& targets) {
    net->train();
    auto optimizer = torch::optim::SGD(net->parameters(), 0.01);
    const int epochs = 5;  // For demonstration, we use 5 epochs.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        auto output = net->forward(inputs);
        auto loss = torch::nn::functional::mse_loss(output, targets);
        loss.backward();
        optimizer.step();
        std::cout << "Epoch " << epoch << " - Loss: " << loss.item<float>() << std::endl;
    }
}

torch::Tensor BaseModel::predict(const torch::Tensor& input) {
    net->eval();
    torch::NoGradGuard no_grad;
    return net->forward(input);
}

void BaseModel::saveModel(const std::string& path) const {
    torch::save(net, path);
}

void BaseModel::loadModel(const std::string& path) {
    torch::load(net, path);
}

std::ostream& operator<<(std::ostream& os, const BaseModel& model) {
    os << "BaseModel: " << model.name << ".\n";
    model.net->pretty_print(os);
    return os;
}
