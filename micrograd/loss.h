#include <memory>
#include "value.h"
#include "ops.h"

// std::shared_ptr<Value> crossEntropyLoss(const std::vector<std::shared_ptr<Value>> &y_pred, const std::shared_ptr<Value> &y_truth);
std::shared_ptr<Value> crossEntropyLoss(const std::vector<std::shared_ptr<Value>> &y_pred, const std::vector<std::shared_ptr<Value>> &y_truth);

std::shared_ptr<Value> mse(const std::vector<std::shared_ptr<Value>> &y_pred, const std::shared_ptr<Value> &y_truth);

std::shared_ptr<Value> svm(const std::vector<std::shared_ptr<Value>> &y_pred, const std::shared_ptr<Value> &y_truth, double margin = 1.0);
