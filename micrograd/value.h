#pragma once
#include <tuple>
#include <memory>
#include <functional>
#include <stack>
#include <unordered_set>

struct Value : public std::enable_shared_from_this<Value>
{
    double data;
    double grad;
    double grad2;
    std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>> children;
    Value(double data);
    void postDFS(
        const std::shared_ptr<Value> &current,
        std::unordered_set<std::shared_ptr<Value>> &visited,
        std::vector<std::shared_ptr<Value>> &result);
    void backward();
    std::function<void()> _backward;
};