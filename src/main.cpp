#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "models/BaseModel.hpp"
#include "data/ImageLoader.hpp"
#include "evaluation/Benchmark.hpp"

int main() {
    // Directories for training and testing
    std::string trainImagesDir = "data/train/images";
    std::string trainMasksDir = "data/train/masks";
    std::string testImagesDir = "data/test/images";
    std::string testMasksDir = "data/test/masks";
    
    // Define target size (e.g., 256x256)
    cv::Size targetSize(256, 256);
    
    // Create ImageLoader instances for images and masks
    ImageLoader trainImageLoader(trainImagesDir, targetSize);
    ImageLoader trainMaskLoader(trainMasksDir, targetSize);
    ImageLoader testImageLoader(testImagesDir, targetSize);
    ImageLoader testMaskLoader(testMasksDir, targetSize);
    
    // Load training data
    std::vector<torch::Tensor> trainImages;
    std::vector<torch::Tensor> trainMasks;
    for (int i = 0; i < 80; ++i) {
        std::string filename = std::to_string(i) + ".png";
        torch::Tensor imgTensor = trainImageLoader.loadCached(filename);
        torch::Tensor maskTensor = trainMaskLoader.loadCached(filename);
        if (imgTensor.defined() && maskTensor.defined()) {
            trainImages.push_back(imgTensor);
            trainMasks.push_back(maskTensor);
        } else {
            std::cerr << "Error loading training image or mask: " << filename << "\n";
        }
    }
    
    // Stack tensors to form a batch (assume all images have the same dimensions)
    torch::Tensor batchImages = torch::stack(trainImages);
    torch::Tensor batchMasks = torch::stack(trainMasks);
    
    // Create a BaseModel instance for segmentation
    BaseModel model("VesselSegmenter", 1, 1);
    std::cout << model << std::endl;
    
    // Train the model on the training batch
    std::cout << "Training model..." << "\n";
    model.trainModel(batchImages, batchMasks);
    
    // Test the model
    Benchmark testbench;
    double accuracy = 0, precision = 0, recall = 0, f1score = 0;
    for (int i = 0; i < 20; ++i) {
        // Load and run inference on test image
        std::string filename = std::to_string(i) + ".png";
        torch::Tensor imgTensor = testImageLoader.loadCached(filename);
        torch::Tensor predTensor = model.predict(imgTensor);

        // Convert to cv::Mat, load test mask and run benchmark
        cv::Mat predMat = testImageLoader.tensorToMat(predTensor);
        cv::Mat maskMat = testMaskLoader.loadRaw(filename);
        accuracy += testbench.computeAccuracyPixels(predMat, maskMat);
        precision += testbench.computePrecisionPixels(predMat, maskMat);
        recall += testbench.computeRecallPixels(predMat, maskMat);
        f1score += testbench.computeF1Pixels(predMat, maskMat);

        // Show images on screen
        cv::imshow("Predicted image", predMat);
        cv::imshow("Ground truth image", maskMat);
        cv::waitKey(0);
    }
    // Compute mean and print results
    accuracy /= 20;
    precision /= 20;
    recall /= 20;
    f1score /= 20;
    std::cout << "Accuracy: " << accuracy << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
    std::cout << "F1 score: " << f1score << "\n";
    return 0;
}
