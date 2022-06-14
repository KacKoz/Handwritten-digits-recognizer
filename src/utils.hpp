#ifndef UTILS_H
#define UTILS_H

#include <cmath>

inline double sigmoid(const double x)
{
    return 1/(1 + exp(-x));
}

#endif