#include "common/Exception.hpp"

med::error::Exception::Exception(std::string message)
: message_(std::move(message)) {}

med::error::Exception::~Exception() noexcept = default;

const char* med::error::Exception::what() const noexcept {
    return message_.c_str();
}

med::error::FileIOException::FileIOException(const std::string& path, bool isRead)
: Exception(std::string("File I/O error ") + (isRead ? "reading " : "writing ") + path) {}

med::error::FileIOException::~FileIOException() noexcept = default;

med::error::DataProcessingException::DataProcessingException(const std::string& stage, const std::string& details)
: Exception("Data processing error at " + stage + ": " + details) {}

med::error::DataProcessingException::~DataProcessingException() noexcept = default;

med::error::ModelException::ModelException(const std::string& details)
: Exception("Model error: " + details) {}

med::error::ModelException::~ModelException() noexcept = default;
