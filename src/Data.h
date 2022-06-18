
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

private:
    std::vector<Eigen::VectorXd> images;
    std::vector<int> expectedDigit;
};