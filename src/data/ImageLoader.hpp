#pragma once

#include "common/Exception.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace med {
namespace data {

class ImageLoader {
public:
    // Constructor 
    ImageLoader(const std::string& imageDir, const cv::Size& targetSize);

    // Loads raw image from given path (relative to imageDir) and returns a cv::Mat
    cv::Mat loadRaw(const std::string& filePath) const;

    // Processes image (resize, convert to grayscale, threshold) and convert to torch::Tensor
    torch::Tensor process(const cv::Mat& img) const;

    // Loads processed image as tensor (if a cache version exists, load it, otherwise process it and save)
    torch::Tensor loadCached(const std::string& filePath);

    // Save processed tensor in a cache file
    void cache(const std::string& filePath, const torch::Tensor& tensor) const;

    // Converts a cv::Mat to torch::Tensor (float, normalized to [0,1])
    torch::Tensor matToTensor(const cv::Mat& img) const;

    // Converts a 1-channel torch::Tensor to cv::Mat
    cv::Mat tensorToMat(const torch::Tensor& tensor) const;

    // Overloaded operator<< for printing loader info
    friend std::ostream& operator<<(std::ostream& os, const ImageLoader& loader) {
        os << "ImageLoader:\n"
           << "  Root directory: " << loader.rootDir << "\n"
           << "  Target size: " << loader.targetSize.width << "x" << loader.targetSize.height << "\n"
           << "  Cache directory: " << loader.cacheDir;
        return os;
    }

private:
    std::string rootDir;   // Directory from which images are loaded
    cv::Size targetSize;   // Target dimension for the resizing step
    std::string cacheDir;  // Directory for processed images caching
};

} // namespace data
} // namespace med
