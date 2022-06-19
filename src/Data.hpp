
#ifndef DATA_H
#define DATA_H

#include "string.h"
#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>

class Data
{

public:
    Data();
    ~Data();
    void readFromFile(const std::string filename);

    std::vector<Eigen::VectorXd> images;
    std::vector<int> expectedDigit;
};

#endif