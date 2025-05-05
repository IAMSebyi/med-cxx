#pragma once

#include <torch/torch.h>

namespace med {

namespace loss {

// Dice loss function
torch::Tensor diceLoss(torch::Tensor preds, torch::Tensor targets);

}

} // namespace med
