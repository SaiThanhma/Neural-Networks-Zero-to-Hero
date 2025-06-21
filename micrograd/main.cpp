#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>

#include "memory"
#include "value.h"
#include "iostream"
#include "nn.h"
#include "ops.h"
#include "optimizer.h"
#include "loss.h"
#include "helper.h"

int main()
{
    std::string path = "names.txt";
    std::ifstream file(path);

    std::string n;
    std::vector<std::string> names;

    while (getline(file, n))
    {
        names.emplace_back(n);
    }

    std::set<char> unique_chars;

    for (const auto &str : names)
    {
        for (char c : str)
        {
            unique_chars.insert(c);
        }
    }

    std::unordered_map<char, int> ctoi;

    ctoi['.'] = 0;
    int i = 1;
    for (auto it = unique_chars.begin(); it != unique_chars.end(); ++it, ++i)
    {
        ctoi[*it] = i;
    }

    std::unordered_map<int, char> itoc;
    for (const auto &pair : ctoi)
    {
        itoc[pair.second] = pair.first;
    }

    std::vector<std::vector<std::shared_ptr<Value>>> x;
    std::vector<std::vector<std::shared_ptr<Value>>> y;
    x.reserve(names.size());
    y.reserve(names.size());

    size_t total_chars = unique_chars.size() + 1;

    for (const auto &name : names)
    {
        std::string n = "." + name + ".";
        for (size_t i = 0; i < n.length() - 2; ++i)
        {
            char char1 = n[i];
            char char2 = n[i + 1];
            char char3 = n[i + 2];

            int index1 = ctoi[char1];
            int index2 = ctoi[char2];
            int index3 = ctoi[char3];

            std::vector<std::shared_ptr<Value>> first = one_hot(std::make_shared<Value>(index1), total_chars);
            std::vector<std::shared_ptr<Value>> second = one_hot(std::make_shared<Value>(index2), total_chars);
            std::vector<std::shared_ptr<Value>> third = one_hot(std::make_shared<Value>(index3), total_chars);

            x.emplace_back(concate(first, second));
            y.emplace_back(third);
        }
    }

    MLP mlp = MLP(total_chars + total_chars, {27}, {id});
    Adam adam = Adam(mlp.parameters(), 0.01, 0.001, 0.9, 0.999, 1e-7);

    std::function<std::shared_ptr<Value>(const std::vector<std::shared_ptr<Value>> &, const std::vector<std::shared_ptr<Value>> &)> loss_fn = crossEntropyLoss;

    int iterations = 100;

    for (int i = 0; i < iterations; ++i)
    {

        std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);
        for (int j = 0; j < 1000; ++j)
        {
            std::vector<std::shared_ptr<Value>> pred = mlp.forward(x.at(j));
            std::vector<std::shared_ptr<Value>> truth = y.at(j);

            loss = loss + loss_fn(pred, truth);
        }
        loss = loss / std::make_shared<Value>(x.size());
        mlp.zero_grad();
        loss->backward();
        adam.step();

        std::cout << "Avg Loss : " << loss->data << std::endl;
    }
}