#pragma once

#include <torch/torch.h>

// Dice loss function
torch::Tensor diceLoss(torch::Tensor preds, torch::Tensor targets);
