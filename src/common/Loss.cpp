#include "Loss.hpp"

torch::Tensor med::loss::diceLoss(torch::Tensor preds, torch::Tensor targets) {
    preds = torch::sigmoid(preds);
    auto intersection = (preds * targets).sum();
    auto union_ = preds.sum() + targets.sum();
    return 1.0 - 2.0 * intersection / (union_ + 1e-6);
}