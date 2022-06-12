#include "Data.cpp"
#include <fstream>
#include <iostream>
#include "stdio.h"

int main(int argc, char *argv[])
{
    std::string trainFile = argv[1];

    Data trainingData;

    trainingData.readFromFile(trainFile);
    return 0;
}