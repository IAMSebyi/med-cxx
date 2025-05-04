#include "BaseModel.hpp"

BaseModel::BaseModel(const std::string& name, torch::Device device)
: name(name), device(device) {}

BaseModel::BaseModel(const BaseModel& other)
: name(other.name), device(other.device) {}

BaseModel& BaseModel::operator=(const BaseModel& other) {
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
    std::cout << "[" << name << "] saved model to " << filename << "\n";
}

void BaseModel::loadModel(const std::string& filename) {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    this->load(archive);
    std::cout << "[" << name << "] saved model to " << filename << "\n";
}

std::ostream& operator<<(std::ostream& os, const BaseModel& model) {
    os << "BaseModel: " << model.name << ".\n";
    return os;
}
