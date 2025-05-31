#include "ImageLoader.hpp"

namespace med {
namespace data {

ImageLoader::ImageLoader(const std::string& imageDir, const cv::Size& targetSize)
    : rootDir(imageDir), targetSize(targetSize)
{
    // Cache directory will be "rootDir/cache"
    cacheDir = rootDir + "/cache";
    if (!fs::exists(cacheDir)) {
        fs::create_directories(cacheDir);
    }
}

cv::Mat ImageLoader::loadRaw(const std::string& filePath) const {
    std::string fullPath = rootDir + "/" + filePath;
    cv::Mat img = cv::imread(fullPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw med::error::FileIOException(fullPath, true);
    }
    return img;
}

torch::Tensor ImageLoader::process(const cv::Mat& img) const {
    cv::Mat resized, gray, thresh;
    try {
        cv::resize(img, resized, targetSize);
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        // Apply Otsu thresholding to obtain a binary image
        cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } catch (const cv::Exception& e) {
        throw med::error::DataProcessingException("process", e.what());
    }

    // Convert processed image to tensor
    torch::Tensor tensor = matToTensor(gray);
    return tensor;
}

torch::Tensor ImageLoader::loadCached(const std::string& filePath) {
    fs::path p(filePath);
    std::string cacheFile = cacheDir + "/" + p.stem().string() + ".pt";

    // If cached image exists, load it; otherwise, process it and cache it.
    if (fs::exists(cacheFile)) {
        torch::Tensor tensor;
        torch::load(tensor, cacheFile);
        return tensor;
    } else {
        cv::Mat raw = loadRaw(filePath);
        if (raw.empty()) {
            throw med::error::FileIOException(cacheFile, true);
        }
        torch::Tensor processed = process(raw);
        cache(filePath, processed);
        return processed;
    }
}

void ImageLoader::cache(const std::string& filePath, const torch::Tensor& tensor) const {
    fs::path p(filePath);
    std::string cacheFile = cacheDir + "/" + p.stem().string() + ".pt";
    try {
        torch::save(tensor, cacheFile);
    } catch (const c10::Error& e) {
        throw med::error::FileIOException(cacheFile, false);
    }
}

torch::Tensor ImageLoader::matToTensor(const cv::Mat& img) const {
    // Convert image to float32 and normalize to [0,1]
    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32F, 1.0 / 255);
    // Create tensor: size [height, width]
    torch::Tensor tensor = torch::from_blob(floatImg.data, {img.rows, img.cols}, torch::kFloat);
    // Add a channel dimension: [1, height, width]
    tensor = tensor.unsqueeze(0);
    return tensor.clone(); // Clone to get data
}

cv::Mat ImageLoader::tensorToMat(const torch::Tensor& tensor) const {
    torch::Tensor cpu = tensor.detach().to(torch::kCPU).squeeze();
    int height = cpu.size(0), width = cpu.size(1);

    if (cpu.dtype() == torch::kUInt8) {
        // Clone the data into a new U8 Mat and scale it in place to 0/255
        cv::Mat mat{height, width, CV_8U, cpu.data_ptr<uint8_t>()};
        mat.convertTo(mat, CV_8U, 255.0);
        return mat.clone();
    }

    // Existing float path
    torch::Tensor f = cpu.to(torch::kFloat);
    return cv::Mat{height, width, CV_32F, f.data_ptr<float>()}.clone();
}

} // namespace data
} // namespace med
