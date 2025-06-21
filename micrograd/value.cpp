#include "value.h"
#include <iostream>

Value::Value(double data) : data{data}, grad{0.0}, children({nullptr, nullptr}) {}

void Value::postDFS(
    const std::shared_ptr<Value> &current,
    std::unordered_set<std::shared_ptr<Value>> &visited,
    std::vector<std::shared_ptr<Value>> &result)
{
    if (current && !visited.count(current))
    {
        visited.insert(current);
        if (current->children.first)
        {
            postDFS(current->children.first, visited, result);
        }
        if (current->children.second)
        {
            postDFS(current->children.second, visited, result);
        }
        result.emplace_back(current);
    }
}

void Value::backward()
{
    this->grad = 1.0;
    this->grad2 = 0.0;
    std::unordered_set<std::shared_ptr<Value>> visited;
    std::vector<std::shared_ptr<Value>> result;
    postDFS(shared_from_this(), visited, result);

    for (size_t i = result.size(); i-- > 0;)
    {
        if (result.at(i)->_backward)
        {
            result.at(i)->_backward();
        }
    }
}