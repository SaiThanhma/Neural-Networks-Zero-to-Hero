#include "nn.h"

void Module::zero_grad()
{
    std::vector<std::shared_ptr<Value>> params = parameters();
    for (size_t i = 0; i < params.size(); ++i)
    {
        params.at(i)->grad = 0.0;
    }
}

Neuron::Neuron(int dim_in, const std::function<std::shared_ptr<Value>(const std::shared_ptr<Value> &)> activation)
{
    this->activation = activation;

    static std::default_random_engine generator(42);

    // Kaiming Initialization

    double bound = std::sqrt(6.0 / dim_in);
    static std::uniform_real_distribution<double> distribution(-bound, bound);
    w.reserve(dim_in);

    for (int i = 0; i < dim_in; ++i)
    {
        double number = distribution(generator);
        w.emplace_back(std::make_shared<Value>(number));
    }

    this->b = std::make_shared<Value>(0.0);
}

std::shared_ptr<Value> Neuron::forward(const std::vector<std::shared_ptr<Value>> &x)
{
    if (x.size() != w.size())
    {
        // std::cerr << "Error: Input size " << x.size() << " doesn't match weight size " << w.size() << std::endl;
    }
    std::shared_ptr<Value> sum = std::make_shared<Value>(0.0);
    for (size_t i = 0; i < w.size(); ++i)
    {
        auto mul = x.at(i) * w.at(i);
        sum = sum + mul;
    }
    sum = sum + b;
    return activation(sum);
}

std::vector<std::shared_ptr<Value>> Neuron::parameters()
{
    auto params = w;
    params.emplace_back(b);
    return params;
}

Layer::Layer(int dim_in, int dim_out, std::function<std::shared_ptr<Value>(const std::shared_ptr<Value> &)> activation)
{
    for (int i = 0; i < dim_out; ++i)
    {
        neurons.emplace_back(dim_in, activation);
    }
}

std::vector<std::shared_ptr<Value>> Layer::forward(const std::vector<std::shared_ptr<Value>> &x)
{
    std::vector<std::shared_ptr<Value>> out;
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        out.emplace_back(neurons.at(i).forward(x));
    }
    return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters()
{
    std::vector<std::shared_ptr<Value>> params;
    size_t total = 0;

    for (size_t i = 0; i < neurons.size(); ++i)
    {
        total += neurons.at(i).parameters().size();
    }
    params.reserve(total);

    for (size_t i = 0; i < neurons.size(); ++i)
    {
        std::vector<std::shared_ptr<Value>> neuron_params = neurons.at(i).parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

MLP::MLP(const int &dim_in, const std::vector<int> &layers, const std::vector<std::function<std::shared_ptr<Value>(const std::shared_ptr<Value> &)>> &activations)
{
    std::vector<std::function<std::shared_ptr<Value>(const std::shared_ptr<Value> &)>> act = activations;
    if (act.empty())
    {
        act.resize(layers.size(), [](std::shared_ptr<Value> i)
                   { return i; });
    }

    this->layers.emplace_back(Layer(dim_in, layers.at(0), act.at(0)));
    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        this->layers.emplace_back(Layer(layers.at(i), layers.at(i + 1), act.at(i + 1)));
    }
}

std::vector<std::shared_ptr<Value>> MLP::forward(const std::vector<std::shared_ptr<Value>> &x)
{
    std::vector<std::shared_ptr<Value>> out = x;

    for (size_t i = 0; i < layers.size(); ++i)
    {
        out = layers.at(i).forward(out);
    }

    return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters()
{
    std::vector<std::shared_ptr<Value>> params;
    size_t total = 0;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        total += layers.at(i).parameters().size();
    }

    params.reserve(total);

    for (size_t i = 0; i < layers.size(); ++i)
    {
        std::vector<std::shared_ptr<Value>> layer_params = layers.at(i).parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}