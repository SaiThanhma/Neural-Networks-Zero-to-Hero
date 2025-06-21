#include "helper.h"
#include <iostream>

std::vector<std::shared_ptr<Value>> one_hot(const std::shared_ptr<Value> &value, const size_t &num_classes)
{
    std::vector<std::shared_ptr<Value>> res;
    res.reserve(num_classes);

    for (size_t i = 0; i < num_classes; ++i)
    {
        res.emplace_back(std::make_shared<Value>(0.0));
    }

    res.at(value->data)->data = 1.0;
    return res;
}

std::vector<std::shared_ptr<Value>> concate(const std::vector<std::shared_ptr<Value>> &a, const std::vector<std::shared_ptr<Value>> &b)
{
    std::vector<std::shared_ptr<Value>> res;
    res.reserve(a.size() + b.size());
    res.insert(res.end(), a.begin(), a.end());
    res.insert(res.end(), b.begin(), b.end());
    return res;
}