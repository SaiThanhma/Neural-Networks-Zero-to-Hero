#pragma once
#include "value.h"

// Binary
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

bool operator>(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

bool operator<(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

bool operator==(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

std::shared_ptr<Value> pow(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

std::shared_ptr<Value> max(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

std::shared_ptr<Value> min(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);

// Unary
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> exp(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> sqrt(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> log(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> id(const std::shared_ptr<Value> &value);

namespace ops
{
    std::shared_ptr<Value> tanh(const std::shared_ptr<Value> &value);
}

std::shared_ptr<Value> relu(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> sigmoid(const std::shared_ptr<Value> &value);

std::shared_ptr<Value> leakyRelu(const std::shared_ptr<Value> &value, const double &alpha = 0.1);
