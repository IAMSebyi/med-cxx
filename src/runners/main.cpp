// src/runners/main.cpp
#include <iostream>
#include <memory>
#include <filesystem>
#include <torch/torch.h>

#include "common/ArgParser.hpp"
#include "common/Exception.hpp"
#include "trainer/SegmentationTrainer.hpp"
#include "trainer/ClassificationTrainer.hpp"
#include "models/UNet.hpp"
#include "models/DenseNet.hpp"
#include "models/ResNet.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        // Parse CLI arguments -> cfg
        auto cfg = med::common::ArgParser::parse(argc, argv);

        // Dump‐all‐fields to stderr/stdout
        std::cout << "> Parsed configuration:\n";
        std::cout << "  modelType      =  " << static_cast<int>(cfg.modelType) << "\n";
        std::cout << "  trainDir       =  \"" << cfg.segTrainDir << "\"\n";
        std::cout << "  testDir        =  \"" << cfg.segTestDir << "\"\n";
        std::cout << "  clsTrainDir    =  \"" << cfg.clsTrainDir << "\"\n";
        std::cout << "  clsTestDir     =  \"" << cfg.clsTestDir << "\"\n";
        std::cout << "  modelName      =  \"" << cfg.modelName << "\"\n";
        std::cout << "  epochs         =  "   << cfg.epochs << "\n";
        std::cout << "  useCUDA        =  "   << (cfg.useCUDA ? "true" : "false") << "\n";
        std::cout << "  bceWeight      =  "   << cfg.bcePosWeight << "\n";
        std::cout << "  skipTraining   =  "   << (cfg.skipTraining ? "true" : "false") << "\n";
        std::cout << "  modelWeights   =  \"" << cfg.modelWeightsPath << "\"\n";
        std::cout << "  deviceStr      =  \"" << cfg.deviceStr << "\"\n";
        std::cout << "  resnetVersion  =  "   << static_cast<int>(cfg.resnetVersion) << "\n";
        std::cout << "  makeVideo      =  "   << (cfg.makeVideo ? "true" : "false") << "\n";
        std::cout << "  videoFPS       =  "   << cfg.videoFPS << "\n";
        std::cout << "  holdFrames     =  "   << cfg.holdFrames << "\n";
        std::cout << "  printBarWidth  =  "   << cfg.printBarWidth << "\n";
        std::cout << "> End of configuration.\n\n";

        // Decide on device
        bool useCuda = (cfg.deviceStr == "cuda");
        if (useCuda && !torch::cuda::is_available()) {
            std::cout << "[INFO] CUDA requested but not available; using CPU.\n";
            useCuda = false;
        }
        torch::Device device = useCuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

        // Create a polymorphic BaseModel shared_ptr
        std::shared_ptr<med::models::BaseModel> model;

        // Build the actual implementation and upcast
        switch (cfg.modelType) {
            case med::common::ModelType::UNet: {
                auto unetImpl = std::make_shared<med::models::UNetImpl>(1, 1, device);
                model = unetImpl;  // upcast to shared_ptr<BaseModel>
                break;
            }

            case med::common::ModelType::DenseNet: {
                std::vector<int> blockCfg {6,12,24,16};
                int growthRate = 32;
                int numClasses = cfg.clsTrainDir.empty() ? 2 : static_cast<int>(cfg.clsTrainDir.size());
                auto dnetImpl = std::make_shared<med::models::DenseNetImpl>(blockCfg, growthRate, 64, numClasses, device);
                model = dnetImpl; // upcast
                break;
            }

            case med::common::ModelType::ResNet: {
                int numClasses = cfg.clsTrainDir.empty() ? 1000 : static_cast<int>(cfg.clsTrainDir.size());
                // Cast common::ResNetVersion → models::ResNet::Version
                auto versionImpl = static_cast<med::models::ResNet::Version>(cfg.resnetVersion);
                auto resImpl = std::make_shared<med::models::ResNet>(versionImpl, numClasses, device);
                model = resImpl;  // upcast
                break;
            }

            default:
                throw med::error::ConfigException("main", "Unknown model type");
        }

        // Load weights
        if (!cfg.modelWeightsPath.empty() && fs::exists(cfg.modelWeightsPath)) {
            std::cout << "[INFO] Loading weights from " << cfg.modelWeightsPath << "\n";
            model->loadModel(cfg.modelWeightsPath);
        }

        // Instantiate Trainer
        std::unique_ptr<med::trainer::BaseTrainer> trainer;
        if (cfg.modelType == med::common::ModelType::UNet) {
            trainer = std::make_unique<med::trainer::SegmentationTrainer>(model, cfg);
        } else {
            trainer = std::make_unique<med::trainer::ClassificationTrainer>(model, cfg);
        }

        // Train (unless skipping) then save
        if (!cfg.skipTraining) {
            std::cout << "[INFO] Starting training...\n";
            trainer->train();

            std::string outModelPath = (cfg.modelName.empty()
                ? (cfg.modelType == med::common::ModelType::UNet 
                ? "unet" : cfg.modelType == med::common::ModelType::DenseNet 
                ? "densenet" : "resnet") : cfg.modelName) + ".pt";

            model->saveModel(outModelPath);
            std::cout << "[INFO] Model saved to " << outModelPath << "\n";
        }

        // Evaluate
        std::cout << "[INFO] Starting evaluation...\n";
        trainer->evaluate();

        return EXIT_SUCCESS;
    }
    catch (const med::error::Exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception& ex) {
        std::cerr << "[ERROR] std::exception: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
