#include "SegmentationTrainer.hpp"

namespace fs = std::filesystem;

namespace med {
namespace trainer {

SegmentationTrainer::SegmentationTrainer(std::shared_ptr<models::BaseModel> model,
                                         const common::Config& cfg)
: BaseTrainer(std::move(model), cfg) {
    // Must have trainDir
    if (cfg.segTrainDir.empty()) {
        throw error::ConfigException("SegmentationTrainer", "Missing --train-dir for UNet");
    }
    loadFileLists();
}

void SegmentationTrainer::loadFileLists() {
    // train dir: should contain subfolders "image" and "mask"
    std::string imgDir = cfg.segTrainDir + "/image";
    std::string mskDir = cfg.segTrainDir + "/mask";
    if (!fs::exists(imgDir) || !fs::exists(mskDir)) {
        throw error::FileIOException(cfg.segTrainDir, "Train directory must have 'image' and 'mask' subfolders");
    }
    for (auto& p : fs::directory_iterator(imgDir)) {
        if (!p.is_regular_file()) continue;
        // Assume mask has same filename in mask dir
        std::string fname = p.path().filename().string();
        if (!fs::exists(mskDir + "/" + fname)) {
            std::cerr << "[WARN] Mask not found for " << fname << ", skipping.\n";
            continue;
        }
        trainImageFiles.push_back(fname);
    }
    std::sort(trainImageFiles.begin(), trainImageFiles.end());

    // If test dir provided, load test images (masks optional)
    if (!cfg.segTestDir.empty()) {
        std::string tImgDir = cfg.segTestDir + "/image";
        if (!fs::exists(tImgDir)) {
            throw error::FileIOException(cfg.segTestDir, "Test directory must have 'image' subfolder");
        }
        for (auto& p : fs::directory_iterator(tImgDir)) {
            if (!p.is_regular_file()) continue;
            testImageFiles.push_back(p.path().filename().string());
        }
        std::sort(testImageFiles.begin(), testImageFiles.end());
    }
}

void SegmentationTrainer::train() {
    data::ImageLoader imgLoader(cfg.segTrainDir + "/image", cv::Size(256,256));
    data::ImageLoader mskLoader(cfg.segTrainDir + "/mask", cv::Size(256,256));

    torch::optim::Adam optimizer = makeOptimizer();
    model->train();

    size_t totalBatches = trainImageFiles.size();
    for (size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
        double epochLoss = 0.0;
        size_t count = 0;

        for (auto& fname : trainImageFiles) {
            auto imgT = imgLoader.loadCached(fname);
            auto mskT = mskLoader.loadCached(fname);
            if (!imgT.defined() || !mskT.defined()) {
                std::cerr << "[WARN] Skipping " << fname << "\n";
                continue;
            }

            // [C,H,W] -> [1,C,H,W]
            auto input = imgT.unsqueeze(0).to(device);
            auto target = mskT.unsqueeze(0).to(device);

            auto output = model->predict(input);

            // Weighted BCE
            auto bce = torch::nn::functional::binary_cross_entropy_with_logits(
                output, target, torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(torch::tensor(cfg.bcePosWeight).to(device))
            );
            // Dice
            auto dice = med::loss::diceLoss(output, target);

            optimizer.zero_grad();
            (bce + dice).backward();
            optimizer.step();

            epochLoss += (bce.item<double>() + dice.item<double>());
            ++count;

            // Print a live progress bar
            printProgress(count, totalBatches);
            std::cout << "  Epoch " << epoch << "/" << cfg.epochs
                      << ", Batch " << count << "/" << totalBatches
                      << ", AvgLoss=" << (epochLoss / std::max<size_t>(1, count)) << "\r";
        }
        std::cout << "\n";  // newline after each epoch
    }
}

void SegmentationTrainer::evaluate() {
    if (testImageFiles.empty()) {
        std::cerr << "[INFO] No test directory provided; skipping evaluation.\n";
        return;
    }

    data::ImageLoader imgLoader(cfg.segTestDir + "/image", cv::Size(256,256));
    data::ImageLoader mskLoader(cfg.segTestDir + "/mask",  cv::Size(256,256));
    eval::Benchmark bench;

    model->eval();
    torch::NoGradGuard no_grad;

    double sumAcc = 0, sumPrec = 0, sumRec = 0, sumF1 = 0, sumIoU = 0, sumMAE = 0, sumHD = 0;
    size_t testCount = 0;

    // Prepare video writer only if requested
    cv::VideoWriter writer;
    if (cfg.makeVideo) {
        int labelH = 50;
        cv::Size targetSz(256,256);
        int demoW = targetSz.width * 3;
        int demoH = targetSz.height + labelH;
        writer.open(
            cfg.modelName + "_demo.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            cfg.videoFPS,
            cv::Size(demoW, demoH)
        );
        if (!writer.isOpened()) {
            throw error::FileIOException(cfg.modelName + "_demo.mp4", true);
        }
    }

    for (auto& fname : testImageFiles) {
        auto imgT = imgLoader.loadCached(fname);
        auto mskRaw = mskLoader.loadRaw(fname); // raw cv::Mat mask (unprocessed)

        if (!imgT.defined()) 
            continue;
        auto input = imgT.unsqueeze(0).to(device);

        auto logits = model->predict(input);
        auto prob = torch::sigmoid(logits).squeeze();
        auto pred = (prob >= 0.5).to(torch::kU8);

        // Convert to cv::Mat
        auto cpuPred = pred.cpu();
        cv::Mat predMat = imgLoader.tensorToMat(cpuPred);

        // Load ground truth mask
        if (mskRaw.empty()) {
            std::cerr << "[WARN] No ground truth mask for " << fname << ", skipping metrics.\n";
            continue;
        }
        cv::Mat gtMask = mskRaw;
        cv::resize(predMat, predMat, gtMask.size(), 0, 0, cv::INTER_NEAREST);
        if (predMat.channels() > 1) 
            cv::cvtColor(predMat, predMat, cv::COLOR_BGR2GRAY);
        if (gtMask.channels() > 1) 
            cv::cvtColor(gtMask, gtMask, cv::COLOR_BGR2GRAY);
        if (predMat.depth() != CV_8U) 
            predMat.convertTo(predMat, CV_8U, 255);
        if (gtMask.depth() != CV_8U)  
            gtMask.convertTo(gtMask, CV_8U, 255);

        // Metrics
        sumAcc += bench.computeAccuracyPixels (predMat, gtMask);
        sumPrec += bench.computePrecisionPixels (predMat, gtMask);
        sumRec += bench.computeRecallPixels (predMat, gtMask);
        sumF1 += bench.computeF1Pixels (predMat, gtMask);
        sumIoU += bench.computeIoUPixels (predMat, gtMask);
        sumMAE += bench.computeMAE (predMat, gtMask);
        sumHD += bench.computeHausdorff (predMat, gtMask);
        ++testCount;

        // Build a demo frame (Original | GT | Pred)
        if (cfg.makeVideo) {
            cv::Mat original = imgLoader.loadRaw(fname);
            cv::resize(original, original, cv::Size(256,256), 0, 0, cv::INTER_LINEAR);

            cv::Mat gtResized, predResized;
            cv::resize(gtMask, gtResized, cv::Size(256,256), 0,0, cv::INTER_NEAREST);
            cv::resize(predMat, predResized, cv::Size(256,256), 0,0, cv::INTER_NEAREST);

            cv::Mat predColor, gtColor;
            cv::cvtColor(predResized, predColor, cv::COLOR_GRAY2BGR);
            cv::cvtColor(gtResized, gtColor, cv::COLOR_GRAY2BGR);

            common::Visualizer::writeSegmentationFrame(writer, original, gtColor, predColor, 50, cfg.holdFrames);
        }
    }

    if (testCount > 0) {
        std::cout << "\n=== Test results over " << testCount << " images ===\n"
                  << "Accuracy : " << (sumAcc  / testCount) << "\n"
                  << "Precision: " << (sumPrec / testCount) << "\n"
                  << "Recall   : " << (sumRec  / testCount) << "\n"
                  << "F1 Score : " << (sumF1 / testCount) << "\n"
                  << "IoU      : " << (sumIoU / testCount) << "\n"
                  << "MAE      : " << (sumMAE / testCount) << "\n"
                  << "Hausdorff: " << (sumHD / testCount) << "\n\n";
    }

    if (cfg.makeVideo) {
        writer.release();
        std::cout << "Segmentation demo video saved.\n";
    }
}

} // namespace trainer
} // namespace med
