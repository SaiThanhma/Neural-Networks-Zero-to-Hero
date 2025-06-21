#include "loss.h"
#include "iostream"

std::shared_ptr<Value> crossEntropyLoss(const std::vector<std::shared_ptr<Value>> &y_pred, const std::vector<std::shared_ptr<Value>> &y_truth)
{
    std::shared_ptr<Value> logits_max = y_pred.at(0);
    for (size_t i = 1; i < y_pred.size(); ++i)
    {
        if (operator>(y_pred.at(i), logits_max))
        {
            logits_max = y_pred.at(i);
        }
    }

    std::vector<std::shared_ptr<Value>> probs = y_pred;
    std::shared_ptr<Value> sum = std::make_shared<Value>(0.0);

    for (std::shared_ptr<Value> &v : probs)
    {
        sum = sum + exp(v - logits_max);
    }

    std::shared_ptr<Value> log_sum = log(sum);

    size_t truth = 0;

    for (size_t i = 0; i < y_pred.size(); ++i)
    {
        if (y_truth.at(i))
        {
            truth = i;
            break;
        }
    }

    std::shared_ptr<Value> pred = y_pred.at(truth);

    return -pred + logits_max + log_sum;
}

std::shared_ptr<Value> svm(const std::vector<std::shared_ptr<Value>> &y_pred, const std::vector<std::shared_ptr<Value>> &y_truth, double margin)
{

    size_t truth = 0;

    for (size_t i = 0; i < y_pred.size(); ++i)
    {
        if (y_truth.at(i))
        {
            truth = i;
            break;
        }
    }

    std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);

    for (size_t i = 0; i < y_pred.size(); ++i)
    {
        if (i == truth)
        {
            continue;
        }

        loss = loss + max(std::make_shared<Value>(0.0), (y_pred.at(i) - y_pred.at(truth)) + std::make_shared<Value>(margin));
    }

    return loss;
}

std::shared_ptr<Value> mse(const std::vector<std::shared_ptr<Value>> &y_pred, const std::vector<std::shared_ptr<Value>> &y_truth)
{
    std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);

    for (size_t i = 0; i < y_pred.size(); ++i)
    {
        loss = loss + (y_pred.at(i) - y_truth.at(i)) * (y_pred.at(i) - y_truth.at(i));
    }

    return loss / std::make_shared<Value>(y_pred.size());
}