#include "BaseModel.hpp"

namespace med {
namespace models {

BaseModel::BaseModel(const std::string& name, torch::Device device)
: name(name), device(device) {}

BaseModel::BaseModel(const BaseModel& other)
: name(other.name), device(other.device) {}

BaseModel& med::models::BaseModel::operator=(const BaseModel& other) {
    if (this != &other) {
        name = other.name;
        device = other.device;
    }
    return *this;
}

BaseModel::~BaseModel() {}

void BaseModel::saveModel(const std::string& filename) const {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(filename);
    std::cout << "[" << name << "] Saved model to " << filename << "\n";
}

void BaseModel::loadModel(const std::string& filename) {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    this->load(archive);
    std::cout << "[" << name << "] Loaded model from " << filename << "\n";
}

} // namespace models
} // namespace med
