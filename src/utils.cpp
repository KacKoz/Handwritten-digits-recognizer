#include "utils.hpp"

Eigen::VectorXd digitVector(int n)
{
    Eigen::VectorXd res(10);
    for (int i = 0; i < 10; i++)
        res[i] = ((i == n) ? 1 : 0);

    return res;
}

std::vector<std::pair<u_int32_t, u_int32_t>> accuracyVector()
{
    std::vector<std::pair<u_int32_t, u_int32_t>> accuracyVector;
    for (int i = 0; i < 10; i++)
    {
        std::pair<int, int> pair(0, 0);
        accuracyVector.push_back(pair);
    }
    return accuracyVector;
}