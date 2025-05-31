#include "ClassificationTrainer.hpp"

namespace fs = std::filesystem;

namespace med {
namespace trainer {

ClassificationTrainer::ClassificationTrainer(std::shared_ptr<models::BaseModel> model, const common::Config& cfg)
: BaseTrainer(std::move(model), cfg) {
    if (cfg.clsTrainDir.empty()) {
        throw error::ConfigException("ClassificationTrainer", "Missing --train-dir for classification");
    }
    // Build class list from subdirectories under clsTrainDir
    for (auto& p : fs::directory_iterator(cfg.clsTrainDir)) {
        if (p.is_directory()) {
            std::string clsName = p.path().filename().string();
            int idx = static_cast<int>(classes.size());
            classes.push_back(clsName);
            classToIdx[clsName] = idx;
        }
    }
    if (classes.empty()) {
        throw error::FileIOException(cfg.clsTrainDir, "No class subdirectories found under train-dir");
    }
    std::sort(classes.begin(), classes.end());
    // Re‐build map in sorted order
    classToIdx.clear();
    for (int i = 0; i < (int)classes.size(); ++i) {
        classToIdx[classes[i]] = i;
    }
}

std::vector<std::pair<std::string,int>> 
ClassificationTrainer::makeFileLabelList(const std::string& rootDir) {
    std::vector<std::pair<std::string,int>> out;
    for (auto& cls : classes) {
        std::string clsPath = rootDir + "/" + cls;
        if (!fs::exists(clsPath) || !fs::is_directory(clsPath)) {
            std::cerr << "[WARN] Class folder not found: " << clsPath << "\n";
            continue;
        }
        int label = classToIdx.at(cls);
        for (auto& f : fs::directory_iterator(clsPath)) {
            if (f.is_regular_file()) {
                out.emplace_back(f.path().filename().string(), label);
            }
        }
    }
    return out;
}

void ClassificationTrainer::train() {
    // Build train list <filename, label>
    auto trainList = makeFileLabelList(cfg.clsTrainDir);

    // If a test‐dir was given, build test list but we only use trainList here
    if (!cfg.clsTestDir.empty()) {
        // (we may evaluate on it later in evaluate())
    }

    data::ImageLoader imgLoader(cfg.clsTrainDir, cv::Size(224,224));
    // We'll assume images are RGB; loader takes raw, then you do matToTensor + normalize
    // Here, factory‐normalize each image to [0,1] and treat as single‐channel or 3‐channel?
    // For brevity, we’ll do: matToTensor yields [1,H,W] -> duplicate to 3 channels:
    // loader.matToTensor yields single‐channel float; we tile it to 3 channels.

    torch::optim::Adam optimizer = makeOptimizer();
    model->train();

    size_t totalBatches = trainList.size();
    for (size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
        double epochLoss = 0.0;
        size_t count = 0;

        for (auto& [fname, label] : trainList) {
            // Find which class folder contains fname
            // Actually, loader.loadRaw expects full path; combine as rootDir/class/fname
            std::string clsName = classes[label];
            std::string fullPath = cfg.clsTrainDir + "/" + clsName + "/" + fname;
            cv::Mat raw = imgLoader.loadRaw(fullPath);
            if (raw.empty()) continue;

            // Preprocess: resize, matToTensor
            auto imgT = imgLoader.process(raw); // [1,H,W] float
            // Expand to 3 channels by repeating
            auto img3 = torch::cat({imgT, imgT, imgT}, 0).unsqueeze(0).to(device); // [1,3,H,W]

            // Create target tensor
            torch::Tensor target = torch::tensor(label, torch::kLong).unsqueeze(0).to(device);

            // Forward pass
            auto logits = model->predict(img3);
            auto loss = torch::nn::functional::cross_entropy(logits, target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epochLoss += loss.item<double>();
            ++count;

            printProgress(count, totalBatches);
            std::cout << "  Epoch " << epoch << "/" << cfg.epochs 
                      << ", Batch " << count << "/" << totalBatches
                      << ", AvgLoss=" << (epochLoss / std::max<size_t>(1, count)) << "\r";
        }
        std::cout << "\n";
    }
}

void ClassificationTrainer::evaluate() {
    if (cfg.clsTestDir.empty()) {
        std::cerr << "[INFO] No test directory provided; skipping classification evaluation.\n";
        return;
    }

    // Build test list
    auto testList = makeFileLabelList(cfg.clsTestDir);
    data::ImageLoader imgLoader(cfg.clsTestDir, cv::Size(224,224));
    eval::Benchmark bench;

    model->eval();
    torch::NoGradGuard no_grad;

    size_t correct = 0, total = 0;
    for (auto& [fname, gtLabel] : testList) {
        // Determine which class folder fname belongs to; same as train
        std::string clsName = classes[gtLabel];
        std::string fullPath = cfg.clsTestDir + "/" + clsName + "/" + fname;
        cv::Mat raw = imgLoader.loadRaw(fullPath);
        if (raw.empty()) continue;

        auto imgT = imgLoader.process(raw);
        auto img3 = torch::cat({imgT, imgT, imgT}, 0).unsqueeze(0).to(device);
        auto logits = model->predict(img3);
        auto pred = logits.argmax(1).item<int>();

        if (pred == gtLabel) 
            ++correct;
        ++total;
    }
    if (total > 0) {
        double acc = static_cast<double>(correct) / total;
        std::cout << "Classification accuracy: " << (100.0 * acc) << "% (" 
                  << correct << "/" << total << ")\n";
    }
}

} // namespace trainer
} // namespace med
