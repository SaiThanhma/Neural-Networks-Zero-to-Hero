#pragma once
#include <memory>
#include <random>
#include "value.h"
#include "ops.h"

struct Module
{
    virtual ~Module() = default;
    void zero_grad();
    virtual std::vector<std::shared_ptr<Value>> parameters() = 0;
};

struct Neuron : public Module
{
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    std::function<std::shared_ptr<Value>(std::shared_ptr<Value>)> activation;

    Neuron(int dim_in, const std::function<std::shared_ptr<Value>(const std::shared_ptr<Value>&)> activation = [](std::shared_ptr<Value> i)
                       { return i; });
    std::shared_ptr<Value> forward(const std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters() override;
};

struct Layer : public Module
{
    std::vector<Neuron> neurons;

    Layer(int dim_in, int dim_out, const std::function<std::shared_ptr<Value>(const std::shared_ptr<Value>&)> activation = [](std::shared_ptr<Value> i)
                                   { return i; });

    std::vector<std::shared_ptr<Value>> forward(const std::vector<std::shared_ptr<Value>> &x);
    std::vector<std::shared_ptr<Value>> parameters() override;
};

struct MLP : public Module
{
    std::vector<Layer> layers;
    MLP(const int &dim_in, const std::vector<int> &layers, const std::vector<std::function<std::shared_ptr<Value>(const std::shared_ptr<Value>&)>> &activations = {});
    std::vector<std::shared_ptr<Value>> forward(const std::vector<std::shared_ptr<Value>> &x);
    std::vector<std::shared_ptr<Value>> parameters() override;
};