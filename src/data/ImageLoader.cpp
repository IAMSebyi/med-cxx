#include "ImageLoader.hpp"

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
        std::cerr << "Error: Could not load image: " << fullPath << std::endl;
    }
    return img;
}

torch::Tensor ImageLoader::process(const cv::Mat& img) const {
    cv::Mat resized;
    cv::resize(img, resized, targetSize);

    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    // Apply Otsu thresholding to obtain a binary image
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

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
            return torch::Tensor(); // Return empty tensor
        }
        torch::Tensor processed = process(raw);
        cache(filePath, processed);
        return processed;
    }
}

void ImageLoader::cache(const std::string& filePath, const torch::Tensor& tensor) const {
    fs::path p(filePath);
    std::string cacheFile = cacheDir + "/" + p.stem().string() + ".pt";
    torch::save(tensor, cacheFile);
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
    torch::Tensor cpuTensor = tensor.detach().to(torch::kCPU);
    cpuTensor = cpuTensor.squeeze(); // remove batch/channel dimensions if necessary
    int height = cpuTensor.size(0);
    int width = cpuTensor.size(1);
    // Assuming tensor is float and values in [0,1]
    cv::Mat mat(height, width, CV_32F, cpuTensor.data_ptr<float>());
    return mat.clone(); // Clone to get data
}

std::ostream& operator<<(std::ostream& os, const ImageLoader& loader) {
    os << "ImageLoader:\n"
       << "  Root directory: " << loader.rootDir << "\n"
       << "  Target size: " << loader.targetSize.width << "x" << loader.targetSize.height << "\n"
       << "  Cache directory: " << loader.cacheDir;
    return os;
}
