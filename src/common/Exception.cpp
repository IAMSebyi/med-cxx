#include "common/Exception.hpp"

namespace med {
namespace error {

Exception::Exception(std::string message)
: message_(std::move(message)) {}

Exception::~Exception() noexcept = default;

const char* Exception::what() const noexcept {
    return message_.c_str();
}

FileIOException::FileIOException(const std::string& path, bool isRead)
: Exception(std::string("File I/O error ") + (isRead ? "reading " : "writing ") + path) {}

FileIOException::~FileIOException() noexcept = default;

DataProcessingException::DataProcessingException(const std::string& stage, const std::string& details)
: Exception("Data processing error at " + stage + ": " + details) {}

DataProcessingException::~DataProcessingException() noexcept = default;

ModelException::ModelException(const std::string& details)
: Exception("Model error: " + details) {}

ModelException::~ModelException() noexcept = default;

ConfigException::ConfigException(const std::string& context, const std::string& details)
: Exception("Configuration error in " + context + ": " + details) {}

ConfigException::~ConfigException() noexcept = default;

} // namespace error
} // namespace med
