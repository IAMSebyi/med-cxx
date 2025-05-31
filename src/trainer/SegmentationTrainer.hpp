#pragma once

#include "BaseTrainer.hpp"
#include "common/Loss.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace med {
namespace trainer {

class SegmentationTrainer : public BaseTrainer {
public:
    SegmentationTrainer(
        std::shared_ptr<models::BaseModel> model,
        const common::Config& cfg
    );

    // Train UNet on (images, masks) from cfg_.segTrainDir
    void train() override;

    // Evaluate on cfg_.segTestDir and optionally write a demo video
    void evaluate() override;

private:
    // Load dataset filenames (pair of <image_filename, mask_filename>) for train/test
    std::vector<std::string> trainImageFiles;
    std::vector<std::string> testImageFiles;

    // Helpers
    void loadFileLists();   // populate trainImageFiles_ and testImageFiles_
};

} // namespace trainer
} // namespace med
