#include "value.h"
#include <string_view>
struct Initializer
{
    // Initializer(gernator)
    void uniform(std::vector<std::shared_ptr<Value>> &params, double a = 0.0, double b = 1.0);
    void normal(std::vector<std::shared_ptr<Value>> &params, double mean = 0.0, double std = 1.0);
    void constant(std::vector<std::shared_ptr<Value>> &params, double constant);

    void xavier_uniform(std::vector<std::shared_ptr<Value>> &params, double gain = 1.0);
    void xavier_normal(std::vector<std::shared_ptr<Value>> &params, double gain = 1.0);
    void kaiming_uniform(std::vector<std::shared_ptr<Value>> &params, double a = 0.0, std::string_view mode = "fan_in", std::string_view nonlinearity = "leaky_relu");
    void kaiming_normal(std::vector<std::shared_ptr<Value>> &params, double a = 0.0, std::string_view mode = "fan_in", std::string_view nonlinearity = "leaky_relu");
};