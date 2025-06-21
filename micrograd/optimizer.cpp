#include "optimizer.h"
#include <iostream>
Optimizer::Optimizer(std::vector<std::shared_ptr<Value>> parameters, double learning_rate, double weight_decay) : parameters(parameters), learning_rate(learning_rate), weight_decay(weight_decay){}

SGD::SGD(std::vector<std::shared_ptr<Value>> parameters, double learning_rate, double weight_decay, double rho) : Optimizer(parameters, learning_rate, weight_decay), rho(rho), velocities(parameters.size(), 0.0){}

void SGD::step()
{
    for (size_t i = 0; i < parameters.size(); ++i)
    {
        velocities.at(i) = rho * velocities.at(i) + parameters.at(i)->grad;
        parameters.at(i)->data -= learning_rate * (velocities.at(i) + weight_decay * parameters.at(i)->data);
    }
}

Nesterov::Nesterov(std::vector<std::shared_ptr<Value>> parameters, double learning_rate, double weight_decay, double rho) : Optimizer(parameters, learning_rate, weight_decay), rho(rho), velocities(parameters.size(), 0.0){}

void Nesterov::step()
{
    for (size_t i = 0; i < parameters.size(); ++i)
    {
        double old_v = velocities.at(i);
        velocities.at(i) = rho * velocities.at(i) - learning_rate * (parameters.at(i)->grad + weight_decay * parameters.at(i)->data);
        parameters.at(i)->data -= rho * old_v - (1.0 + rho) * velocities.at(i);

    }
}

AdaGrad::AdaGrad(std::vector<std::shared_ptr<Value>> parameters, double learning_rate, double weight_decay, double epsilon) : Optimizer(parameters, learning_rate, weight_decay), epsilon(epsilon), grad_squared(parameters.size(), 0.0){}

void AdaGrad::step()
{
    for (size_t i = 0; i < parameters.size(); ++i)
    {
        grad_squared.at(i) += parameters.at(i)->grad * parameters.at(i)->grad;
        parameters.at(i)->data -= learning_rate * (parameters.at(i)->grad / (std::sqrt(grad_squared.at(i)) + epsilon) + weight_decay * parameters.at(i)->data);
    }
}

RMSProp::RMSProp(std::vector<std::shared_ptr<Value>> parameters, double learning_rate, double weight_decay, double decay_rate, double epsilon) : Optimizer(parameters, learning_rate, weight_decay), decay_rate(decay_rate), epsilon(epsilon), grad_squared(parameters.size(), 0.0){}

void RMSProp::step()
{
    for (size_t i = 0; i < parameters.size(); ++i)
    {
        grad_squared.at(i) = decay_rate * grad_squared.at(i) + (1.0 - decay_rate) * parameters.at(i)->grad * parameters.at(i)->grad;
        parameters.at(i)->data -= learning_rate * (parameters.at(i)->grad / (std::sqrt(grad_squared.at(i)) + epsilon) + weight_decay * parameters.at(i)->data);
    }
}


Adam::Adam(std::vector<std::shared_ptr<Value>> parameters, double learning_rate, double weight_decay, double beta1, double beta2, double epsilon) : Optimizer(parameters, learning_rate, weight_decay), epsilon(epsilon), beta1(beta1), beta2(beta2),moment1(parameters.size(), 0.0), moment2(parameters.size(), 0.0), t(1){}
void Adam::step()
{
    for (size_t i = 0; i < parameters.size(); ++t, ++i)
    {
        moment1.at(i) = beta1 * moment1.at(i) + (1.0 - beta1) * parameters.at(i)->grad;
        moment2.at(i) = beta2 * moment2.at(i) + (1.0 - beta2) * parameters.at(i)->grad * parameters.at(i)->grad;
        double moment_unbias1 = moment1.at(i) / (1.0 - std::pow(beta1, t));
        double moment_unbias2 = moment2.at(i) / (1.0 - std::pow(beta2, t));
        parameters.at(i)->data -= learning_rate * (moment_unbias1 / (std::sqrt(moment_unbias2) + epsilon) + weight_decay * parameters.at(i)->data);
    }
}