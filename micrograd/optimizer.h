#include "value.h"
struct Optimizer
{
    std::vector<std::shared_ptr<Value>> parameters;
    double learning_rate;
    double weight_decay;

    Optimizer(std::vector<std::shared_ptr<Value>> parameters, double learning_rate = 0.01, double weight_decay = 0.0);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
};

struct SGD : public Optimizer{
    double rho;
    std::vector<double> velocities;
    SGD(std::vector<std::shared_ptr<Value>> parameters, double learning_rate = 0.01, double weight_decay = 0.0, double rho = 0.0);
    void step() override;
};

struct Nesterov : public Optimizer{
    double rho;
    std::vector<double> velocities;
    Nesterov(std::vector<std::shared_ptr<Value>> parameters, double learning_rate = 0.01, double weight_decay = 0.0, double rho = 0.0);
    void step() override;
};

struct AdaGrad : public Optimizer {
    double epsilon;
    std::vector<double> grad_squared;
    AdaGrad(std::vector<std::shared_ptr<Value>> parameters, double learning_rate = 0.01, double weight_decay = 0.0, double epsilon = 1e-7);
    void step() override;
};

struct RMSProp : public Optimizer {
    double decay_rate;
    double epsilon;
    std::vector<double> grad_squared;
    RMSProp(std::vector<std::shared_ptr<Value>> parameters, double learning_rate = 0.01, double weight_decay = 0.0, double decay_rate = 0.99, double epsilon = 1e-7);
    void step() override;
};

struct Adam : public Optimizer{
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::vector<double> moment1;
    std::vector<double> moment2;
    Adam(std::vector<std::shared_ptr<Value>> parameters, double learning_rate = 0.01, double weight_decay = 0.0, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
    void step() override;
};