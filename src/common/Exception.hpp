#pragma once

#include <exception>
#include <string>

namespace med {

namespace error {

// Base class for all MED-CXX exceptions
class Exception : public std::exception {
public:
    explicit Exception(std::string message);
    ~Exception() noexcept override;

    // Returns a human-readable description of the error
    const char* what() const noexcept override;

private:
    std::string message_;
};
    
// Thrown on file I/O failures (reading or writing)
class FileIOException : public Exception {
public:
    FileIOException(const std::string& path, bool isRead = true);
    ~FileIOException() noexcept override;
};

// Thrown on errors during data preprocessing (e.g. image resize, threshold)
class DataProcessingException : public Exception {
public:
    DataProcessingException(const std::string& stage, const std::string& details);
    ~DataProcessingException() noexcept override;
};

// Thrown on model‐related errors (loading, saving, mismatched dims, etc.)
class ModelException : public Exception {
public:
    explicit ModelException(const std::string& details);
    ~ModelException() noexcept override;
};

} // namespace error

} // namespace med
