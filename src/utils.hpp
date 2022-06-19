#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <utility>

inline double sigmoid(const double x)
{
    return 1 / (1 + exp(-x));
}

Eigen::VectorXd digitVector(int n);

std::vector<std::pair<u_int32_t, u_int32_t>> accuracyVector();

#endif