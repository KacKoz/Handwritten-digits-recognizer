#include "Data.cpp"
#include <fstream>
#include <iostream>
#include "stdio.h"
#include <iostream>
#include "Layer.hpp"

int main(int argc, char *argv[])
{
    std::string trainFile = argv[1];

    Data trainingData;

    trainingData.readFromFile(trainFile);
    double lr = 0.01;
    Layer input(28*28, 0, LayerType::input, lr);
    Layer hidden(15, 0, LayerType::hidden, lr);
    Layer output(10, 0, LayerType::output, lr);
    
    input.connect(&hidden);
    hidden.connect(&output);

    return 0;
}