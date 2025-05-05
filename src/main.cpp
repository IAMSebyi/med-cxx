#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "common/Exception.hpp"
#include "common/Loss.hpp"
#include "common/Utils.hpp"
#include "data/ImageLoader.hpp"
#include "evaluation/Benchmark.hpp"
#include "models/UNet.hpp"

namespace fs = std::filesystem;

int main() {
    // Blood Vessels Segmentation with UNet - DEMO

    try {
        // Device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << "\n";

        // Image paths
        const std::string trainImagesDir = "data/train/image";
        const std::string trainMasksDir = "data/train/mask";
        const std::string testImagesDir = "data/test/image";
        const std::string testMasksDir = "data/test/mask";

        // ImageLoader (resizes + binarizes masks)
        cv::Size targetSize(256, 256);
        med::data::ImageLoader trainImgLoader(trainImagesDir, targetSize);
        med::data::ImageLoader trainMaskLoader(trainMasksDir, targetSize);
        med::data::ImageLoader testImgLoader(testImagesDir, targetSize);
        med::data::ImageLoader testMaskLoader(testMasksDir, targetSize);

        // Collect file names
        std::vector<std::string> trainFiles, testFiles;
        for (auto& p : fs::directory_iterator(trainImagesDir)) {
            if (p.is_regular_file()) {
                trainFiles.push_back(p.path().filename().string());
            }
        }
        for (auto& p : fs::directory_iterator(testImagesDir)) {
            if (p.is_regular_file()) { 
                testFiles.push_back(p.path().filename().string());
            }
        }

        std::sort(trainFiles.begin(), trainFiles.end());
        std::sort(testFiles.begin(),  testFiles.end());

        // Model
        med::models::UNet model (1, 1, device);
        model->to(device);

        // Check if model already exists
        std::string modelFile = "unet_blood_vessels_segmentation.pt";
        if (fs::exists(modelFile)) {
            std::cout << "Loading existing model from " << modelFile << "\n";
            model->loadModel(modelFile);
        } else {
            std::cout << "No existing model found. Training a new one...\n";
            model->train();

            // Hyperparameters
            const size_t epochs = 200;
            const double lr = 1e-4;
            const double weightBCE = 3.0;

            // Optimizer
            torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));

            // Training loop
            for (size_t epoch = 1; epoch <= epochs; ++epoch) {
                double epoch_loss = 0.0;
                size_t count = 0;

                for (auto& fname : trainFiles) {
                    auto img = trainImgLoader.loadCached(fname);
                    auto mask = trainMaskLoader.loadCached(fname);
                    if (!img.defined() || !mask.defined()) {
                        std::cerr << "[WARN] skipping " << fname << "\n";
                        continue;
                    }

                    // [C, H, W] -> [1, C, H, W]
                    auto input = img.unsqueeze(0).to(device);
                    auto target = mask.unsqueeze(0).to(device);

                    // Forward pass
                    auto output = model->predict(input);

                    // Weighted BCE loss
                    auto bce = torch::nn::functional::binary_cross_entropy_with_logits(
                        output, target, torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
                        .pos_weight(torch::tensor(weightBCE).to(device)));

                    // Dice loss
                    auto dice = med::loss::diceLoss(output, target);

                    optimizer.zero_grad();
                    bce.backward();
                    optimizer.step();

                    epoch_loss += 0.5 * bce.item<double>() + 0.5 * dice.item<double>();
                    ++count;       
                }

                med::util::printProgressBar(epoch, epochs, 50);
                std::cout << "  Epoch: " << epoch << ", Average Loss: " << (epoch_loss / std::max<size_t>(1,count)) << ".\r";
            }
        }

        // Save model
        std::cout << "\n";
        model->saveModel(modelFile);
        std::cout << "Model saved to " << modelFile << "\n";

        // Testing
        model->eval();
        torch::NoGradGuard no_grad;
        med::eval::Benchmark bench;

        double sumAcc = 0, sumPrec = 0, sumRec = 0, sumF1 = 0, sumIoU = 0, sumMAE = 0, sumHD = 0;
        size_t testCount = 0;

        // Create demo video
        // Video settings
        int demoWidth = targetSize.width * 3;
        int demoHeight = targetSize.height;
        int labelHeight = 50;
        int fps = 1;
        cv::VideoWriter writer(
            "demo_blood_vessels_seg_unet.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,
            cv::Size(demoWidth, demoHeight + labelHeight)
        );
        if (!writer.isOpened()) {
            throw med::error::FileIOException("demo_blood_vessels_seg_unet.mp4", true);
        }

        for (auto& fname : testFiles) {
            auto img = testImgLoader.loadCached(fname);
            if (!img.defined()) continue;
            auto input = img.unsqueeze(0).to(device);

            auto logits = model->predict(input);
            auto prob = torch::sigmoid(logits).squeeze();
            auto pred = (prob >= 0.5).to(torch::kU8);

            // Convert to cv::Mat
            auto cpuTensor = pred.cpu();
            cv::Mat predMat = testImgLoader.tensorToMat(cpuTensor);

            // Load ground truth
            cv::Mat maskMat = testMaskLoader.loadRaw(fname);
            cv::resize(predMat, predMat, maskMat.size(), 0, 0, cv::INTER_NEAREST);
            if (predMat.channels() > 1) cv::cvtColor(predMat, predMat, cv::COLOR_BGR2GRAY);
            if (maskMat.channels() > 1) cv::cvtColor(maskMat, maskMat, cv::COLOR_BGR2GRAY);
            if (predMat.depth() != CV_8U) predMat.convertTo(predMat, CV_8U, 255);
            if (maskMat.depth() != CV_8U) maskMat.convertTo(maskMat, CV_8U, 255);

            // Metrics
            sumAcc += bench.computeAccuracyPixels (predMat, maskMat);
            sumPrec += bench.computePrecisionPixels (predMat, maskMat);
            sumRec += bench.computeRecallPixels (predMat, maskMat);
            sumF1 += bench.computeF1Pixels (predMat, maskMat);
            sumIoU += bench.computeIoUPixels(predMat, maskMat);
            sumMAE += bench.computeMAE(predMat, maskMat);
            sumHD += bench.computeHausdorff(predMat, maskMat);
            ++testCount;

            // Write demo video
            // Load original image
            cv::Mat original = testImgLoader.loadRaw(fname);

            // Resize images to target size
            cv::resize(original, original, targetSize, 0, 0, cv::INTER_LINEAR);
            cv::resize(maskMat, maskMat, targetSize, 0, 0, cv::INTER_NEAREST);
            cv::resize(predMat, predMat, targetSize, 0, 0, cv::INTER_NEAREST);

            // Convert ground truth mask and prediction mask images to BGR
            cv::Mat gtColor, predColor;
            cv::cvtColor(maskMat, gtColor, cv::COLOR_GRAY2BGR);
            cv::cvtColor(predMat, predColor, cv::COLOR_GRAY2BGR);

            // Extend the upper part of each image to add a label
            cv::Mat originalWithLabel(targetSize.height + labelHeight, targetSize.width, original.type(), cv::Scalar(0, 0, 0));
            cv::Mat gtColorWithLabel(targetSize.height + labelHeight, targetSize.width, gtColor.type(), cv::Scalar(0, 0, 0));
            cv::Mat predColorWithLabel(targetSize.height + labelHeight, targetSize.width, predColor.type(), cv::Scalar(0, 0, 0));

            // Copy the original images into the lower part of the extended images
            original.copyTo(originalWithLabel(cv::Rect(0, labelHeight, targetSize.width, targetSize.height)));
            gtColor.copyTo(gtColorWithLabel(cv::Rect(0, labelHeight, targetSize.width, targetSize.height)));
            predColor.copyTo(predColorWithLabel(cv::Rect(0, labelHeight, targetSize.width, targetSize.height)));

            // Add text labels in the middle of the upper part
            cv::putText(originalWithLabel, "Original image", cv::Point(targetSize.width / 2 - 50, labelHeight / 2 + 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(gtColorWithLabel, "Ground truth mask", cv::Point(targetSize.width / 2 - 70, labelHeight / 2 + 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(predColorWithLabel, "Predicted mask", cv::Point(targetSize.width / 2 - 60, labelHeight / 2 + 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            // Construct frame with extended height
            std::vector<cv::Mat> panels = { originalWithLabel, gtColorWithLabel, predColorWithLabel };
            cv::Mat frame;
            cv::hconcat(panels, frame);

            // Write frame to video (2 seconds per image)
            for (int i = 0; i < 2; ++i) {
                writer.write(frame);
            }
        }

        if (testCount) {
            std::cout << "\n=== Test results over " << testCount << " images ===\n"
                      << "Accuracy : " << (sumAcc / testCount) << "\n"
                      << "Precision: " << (sumPrec / testCount) << "\n"
                      << "Recall   : " << (sumRec / testCount) << "\n"
                      << "F1 Score : " << (sumF1 / testCount) << "\n"
                      << "IoU      : " << (sumIoU / testCount) << "\n"
                      << "MAE      : " << (sumMAE / testCount) << "\n"
                      << "Hausdorff: " << (sumHD / testCount) << "\n\n";
        }

        writer.release();
        cv::destroyAllWindows();
        std::cout << "Demo video saved as demo_blood_vessels_seg_unet.mp4\n";

        return 0;
    } catch (const med::error::Exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
