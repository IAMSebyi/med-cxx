// common/ArgParser.hpp
#pragma once

#include <string>
#include <vector>

namespace med {
namespace common {

// Which model to run
enum class ModelType { UNet, DenseNet, ResNet, Unknown };

// Which ResNet version (if ModelType::ResNet)
enum class ResNetVersion { R18, R34, R50, R101, R152 };

struct Config {
    // Global
    ModelType modelType = ModelType::Unknown;
    std::string modelName = "";
    std::string modelWeightsPath = "";
    bool skipTraining = false;

    // Device
    bool useCUDA = false;
    std::string deviceStr = "cpu"; // "cpu" or "cuda"

    // Common hyperparameters
    size_t epochs = 50;
    double learningRate = 1e-3;

    // Segmentation‐specific
    std::string segTrainDir = ""; // path to train/images & train/masks
    std::string segTestDir = ""; // path to test/images & optional test/masks
    double bcePosWeight = 1.0; // for weighted BCE

    // Classification‐specific (DenseNet/ResNet)
    std::string clsTrainDir = "";
    std::string clsTestDir = "";
    ResNetVersion resnetVersion = ResNetVersion::R18;

    // Visualizer options
    bool makeVideo = true;
    int videoFPS = 1;
    int holdFrames = 2; // how many frames per sample

    // Miscellaneous
    size_t printBarWidth = 50;
};

//
// A very minimal parser: expects arguments in the form:
//   medcxx <model> [--train-dir PATH] [--test-dir PATH]
//           [--model-name NAME] [--weights path]
//           [--skip-training] [--cuda]
//           [--epochs N] [--lr LR] [--bce-weight W]
//           [--resnet-version R18|R34|R50|R101|R152]
//           [--no-video] [--fps N] [--hold N]
//  

class ArgParser {
public:
    static Config parse(int argc, char** argv);
};

} // namespace common
} // namespace med
