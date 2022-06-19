#include "utils.hpp"

Eigen::VectorXd digitVector(int n)
{
    Eigen::VectorXd res(10);
    for(int i=0; i<10; i++)
        res[i] = ((i==n) ? 1 : 0);
    
    return res;
}