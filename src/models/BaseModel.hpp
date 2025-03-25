#pragma once

#include <torch/torch.h>
#include <iostream>
#include <string>

// BaseModel is a simple CNN model for demonstration purposes
// It supports training, prediction, saving, and loading
class BaseModel {
public:
    // Constructor with parameters
    BaseModel(const std::string& modelName, int inChannels = 1, int outChannels = 2);

    // Copy constructor (performs deep copy via clone)
    BaseModel(const BaseModel& other);

    // Copy assignment operator (performs deep copy via clone)
    BaseModel& operator=(const BaseModel& other);

    // Destructor
    virtual ~BaseModel();

    // Train the model with input and target tensors (using a simple MSE loss for demonstration)
    virtual void trainModel(const torch::Tensor& inputs, const torch::Tensor& targets);

    // Predict output given an input tensor
    virtual torch::Tensor predict(const torch::Tensor& input);

    // Save the model's state to a file
    virtual void saveModel(const std::string& path) const;

    // Load the model's state from a file
    virtual void loadModel(const std::string& path);

    // Overloaded operator<< for printing model info
    friend std::ostream& operator<<(std::ostream& os, const BaseModel& model);

protected:
    std::string name;
    // Using torch::nn::Sequential to build a simple CNN
    torch::nn::Sequential net;
    torch::Device device;
};
