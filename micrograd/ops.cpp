#include "ops.h"
#include <iostream> 

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    auto res = std::make_shared<Value>(a->data + b->data);
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        a->grad += res->grad;
        b->grad += res->grad;
    };

    return res;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    auto res = std::make_shared<Value>(a->data * b->data);
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        a->grad += res->grad * b->data;
        b->grad += res->grad * a->data;

        a->grad2 += res->grad2 * b->data * b->data + 2.0 * res->grad * b->grad;
        b->grad2 += res->grad2 * a->data * a->data + 2.0 * res->grad * a->grad;
                 
    };

    return res;
}

std::shared_ptr<Value> pow(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    auto res = std::make_shared<Value>(std::pow(a->data, b->data));
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        
        a->grad += res->grad * b->data * std::pow(a->data, b->data - 1.0) ;
        b->grad += res->grad * res->data * std::log(a->data);
    };

    return res;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    auto res = std::make_shared<Value>(a->data / b->data);
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        a->grad += res->grad / b->data;
        b->grad -= res->grad * a->data / (b->data * b->data);
    };

    return res;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    auto res = std::make_shared<Value>(a->data - b->data);
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        a->grad += res->grad;
        b->grad -= res->grad;
    };

    return res;
}

bool operator>(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b){
    return a->data > b->data;
}

bool operator<(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b){
    return a->data < b->data;
}

bool operator==(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b){
    return a->data == b->data;
}

std::shared_ptr<Value> max(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b){
    auto res = std::make_shared<Value>(a->data > b->data ? a->data : b->data);
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        if(a->data > b->data){
            a->grad += res->grad;
            b->grad += 0;
        }
        else{
            a->grad += 0.0;
            b->grad += res->grad;
        }
    };

    return res;
}

std::shared_ptr<Value> min(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b){
    auto res = std::make_shared<Value>(a->data > b->data ? b->data: a->data);
    res->children = {a, b};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        const auto &b = res->children.second;
        if(a->data > b->data){
            a->grad += 0.0;
            b->grad += res->grad;
        }
        else{
            a->grad += res->grad;
            b->grad += 0.0;
        }
    };

    return res;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>&value){
    auto res = std::make_shared<Value>(value->data);
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad;
    };

    return res;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>&value){
    auto res = std::make_shared<Value>(-value->data);
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad -= res->grad;
    };

    return res;
}

std::shared_ptr<Value> sqrt(const std::shared_ptr<Value> &value){
    auto res = std::make_shared<Value>(std::sqrt(value->data));
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad /(2*res->data);
    };

    return res;
}

std::shared_ptr<Value> exp(const std::shared_ptr<Value> &value){
    auto res = std::make_shared<Value>(std::exp(value->data));
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad * res->data;
    };

    return res;
}


std::shared_ptr<Value> log(const std::shared_ptr<Value> &value){
    auto res = std::make_shared<Value>(std::log(value->data));
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad / a->data;
    };

    return res;
}

std::shared_ptr<Value> id(const std::shared_ptr<Value> &value){
    auto res = std::make_shared<Value>(value->data);
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad;
    };

    return res;
}

std::shared_ptr<Value> sigmoid(const std::shared_ptr<Value> &value){
    auto res = std::make_shared<Value>(1.0 / (1.0 + std::exp(-value->data)));
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad * res->data * (1 - res->data);
    };

    return res;
}

std::shared_ptr<Value> ops::tanh(const std::shared_ptr<Value> &value){
    auto res = std::make_shared<Value>(std::tanh(value->data));
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad * (1 - res->data * res->data);
    };

    return res;
}

std::shared_ptr<Value> relu(const std::shared_ptr<Value> &value){
    std::shared_ptr<Value> res = std::make_shared<Value>((value->data < 0.0) ? 0.0 : value->data);
    res->children = {value, nullptr};

    res->_backward = [res]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad * ((a->data < 0) ? 0.0 : 1.0);
    };
    return res;
}

std::shared_ptr<Value> leakyRelu(const std::shared_ptr<Value> &value, const double &alpha){
    std::shared_ptr<Value> res = std::make_shared<Value>((value->data < 0.0) ? alpha * value->data : value->data);
    res->children = {value, nullptr};

    res->_backward = [res, alpha]()
    {
        const auto &a = res->children.first;
        a->grad += res->grad * ((a->data < 0.0) ? alpha : 1.0);
    };
    return res;
}