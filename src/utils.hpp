#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <Eigen/Dense>

inline double sigmoid(const double x)
{
    return 1/(1 + exp(-x));
}

Eigen::VectorXd digitVector(int n);

#endif