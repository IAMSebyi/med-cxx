// common/ArgParser.cpp
#include "ArgParser.hpp"
#include <iostream>
#include <algorithm>

namespace med {
namespace common {

// Convert string to lowercase
static std::string toLower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

// Parse ResNet version
static ResNetVersion parseResNetVersion(const std::string& s) {
    std::string low = toLower(s);
    if (low == "r18")  return ResNetVersion::R18;
    if (low == "r34")  return ResNetVersion::R34;
    if (low == "r50")  return ResNetVersion::R50;
    if (low == "r101") return ResNetVersion::R101;
    if (low == "r152") return ResNetVersion::R152;
    return ResNetVersion::R18; // default
}

Config ArgParser::parse(int argc, char** argv) {
    Config cfg;

    if (argc < 2) {
        std::cerr << "Usage: medcxx <model> [options]\n"
                  << "  <model>: unet | densenet | resnet\n"
                  << "Options:\n"
                  << "  --train-dir <path>       Path to training data\n"
                  << "  --test-dir <path>        Path to test data\n"
                  << "  --model-name <name>      Human‐readable name (prefixed by model)\n"
                  << "  --weights <path>         Path to .pt weights (load & skip training)\n"
                  << "  --skip-training          Skip training entirely\n"
                  << "  --cuda                   Use CUDA if available\n"
                  << "  --epochs, -e <N>         Number of epochs (default 50)\n"
                  << "  --lr, -l <LR>            Learning rate (default 1e-3)\n"
                  << "  --bce-weight <W>         BCE positive weight (segmentation)\n"
                  << "  --resnet-version <VER>   R18|R34|R50|R101|R152 (default R18)\n"
                  << "  --no-video               Disable writing a demo video\n"
                  << "  --fps <N>                FPS for video (default 1)\n"
                  << "  --hold <N>               Frames to hold each sample (default 2)\n\n";
        std::exit(EXIT_FAILURE);
    }

    // Model type
    std::string modelStr = toLower(argv[1]);
    if (modelStr == "unet")         
        cfg.modelType = ModelType::UNet;
    else if (modelStr == "densenet")
        cfg.modelType = ModelType::DenseNet;
    else if (modelStr == "resnet")  
        cfg.modelType = ModelType::ResNet;
    else {
        std::cerr << "[ERROR] Unknown model: " << argv[1] << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Scan the rest of argv for options
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--train-dir" && i+1 < argc) {
            if (cfg.modelType == ModelType::UNet)     
                cfg.segTrainDir = argv[++i];
            else                                      
                cfg.clsTrainDir = argv[++i];
        }
        else if (arg == "--test-dir" && i+1 < argc) {
            if (cfg.modelType == ModelType::UNet)     
                cfg.segTestDir = argv[++i];
            else                                      
                cfg.clsTestDir = argv[++i];
        }
        else if ((arg == "--model-name" || arg == "-n") && i+1 < argc) {
            cfg.modelName = argv[++i];
        }
        else if ((arg == "--weights" || arg == "-w") && i+1 < argc) {
            cfg.modelWeightsPath = argv[++i];
        }
        else if (arg == "--skip-training") {
            cfg.skipTraining = true;
        }
        else if (arg == "--cuda") {
            cfg.useCUDA = true;
            cfg.deviceStr = "cuda";
        }
        else if ((arg == "--epochs" || arg == "-e") && i+1 < argc) {
            cfg.epochs = static_cast<size_t>(std::stoul(argv[++i]));
        }
        else if ((arg == "--lr" || arg == "-l") && i+1 < argc) {
            cfg.learningRate = std::stod(argv[++i]);
        }
        else if ((arg == "--bce-weight") && i+1 < argc) {
            cfg.bcePosWeight = std::stod(argv[++i]);
        }
        else if ((arg == "--resnet-version") && i+1 < argc) {
            cfg.resnetVersion = parseResNetVersion(argv[++i]);
        }
        else if (arg == "--no-video") {
            cfg.makeVideo = false;
        }
        else if ((arg == "--fps") && i+1 < argc) {
            cfg.videoFPS = std::stoi(argv[++i]);
        }
        else if ((arg == "--hold") && i+1 < argc) {
            cfg.holdFrames = std::stoi(argv[++i]);
        }
        else if ((arg == "--help") || (arg == "-h")) {
            std::cout << "Usage: medcxx <model> [options]\n"
                      << "  <model>: unet | densenet | resnet\n"
                      << "Options:\n"
                      << "  --train-dir <path>       Path to training data\n"
                      << "  --test-dir <path>        Path to test data\n"
                      << "  --model-name <name>      Human‐readable name (prefixed by model)\n"
                      << "  --weights <path>         Path to .pt weights (load & skip training)\n"
                      << "  --skip-training          Skip training entirely\n"
                      << "  --cuda                   Use CUDA if available\n"
                      << "  --epochs, -e <N>         Number of epochs (default 50)\n"
                      << "  --lr, -l <LR>            Learning rate (default 1e-3)\n"
                      << "  --bce-weight <W>         BCE positive weight (segmentation)\n"
                      << "  --resnet-version <VER>   R18|R34|R50|R101|R152 (default R18)\n"
                      << "  --no-video               Disable writing a demo video\n"
                      << "  --fps <N>                FPS for video (default 1)\n"
                      << "  --hold <N>               Frames to hold each sample (default 2)\n"
                      << std::endl;
            std::exit(EXIT_SUCCESS);
        }
        else {
            std::cerr << "[WARN] Unknown option: " << arg << "\n";
        }
    }

    return cfg;
}

} // namespace common
} // namespace med
