#include "BaseModel.hpp"

med::models::BaseModel::BaseModel(const std::string& name, torch::Device device)
: name(name), device(device) {}

med::models::BaseModel::BaseModel(const BaseModel& other)
: name(other.name), device(other.device) {}

med::models::BaseModel& med::models::BaseModel::operator=(const BaseModel& other) {
    if (this != &other) {
        name = other.name;
        device = other.device;
    }
    return *this;
}

med::models::BaseModel::~BaseModel() {}

void med::models::BaseModel::saveModel(const std::string& filename) const {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(filename);
    std::cout << "[" << name << "] saved model to " << filename << "\n";
}

void med::models::BaseModel::loadModel(const std::string& filename) {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    this->load(archive);
    std::cout << "[" << name << "] saved model to " << filename << "\n";
}
