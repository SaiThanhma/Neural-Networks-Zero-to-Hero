#include <value.h>

std::vector<std::shared_ptr<Value>> one_hot(const std::shared_ptr<Value> &value, const size_t &num_classes);

std::vector<std::shared_ptr<Value>> concate(const std::vector<std::shared_ptr<Value>> &a, const std::vector<std::shared_ptr<Value>> &b);