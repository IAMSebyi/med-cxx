#pragma once

#include <torch/torch.h>
#include <iostream>
#include <string>

namespace med {

namespace models {

// Base model class for all models
class BaseModel : public torch::nn::Module {
public:
    // Constructor with parameters
    BaseModel(const std::string& name, torch::Device device = torch::kCPU);

    // Copy constructor (performs deep copy via clone)
    BaseModel(const BaseModel& other);

    // Copy assignment operator (performs deep copy via clone)
    BaseModel& operator=(const BaseModel& other);

    // Destructor
    virtual ~BaseModel();

    // Predict output given an input tensor
    virtual torch::Tensor predict(const torch::Tensor& input) = 0;

    // Save model weights to file
    void saveModel(const std::string& filename) const;

    // Load model weights from file
    void loadModel(const std::string& filename);

    // Overloaded operator<< for printing model info
    friend std::ostream& operator<<(std::ostream& os, const BaseModel& model) {
        os << model.name << " model on device " << model.device;
        return os;
    }

protected:
    std::string name;
    torch::Device device;
};

} // namespace models

} // namespace med
