#include "Data.hpp"
#include <fstream>
#include <iostream>
#include "stdio.h"
#include <iostream>
#include "Neuralnet.hpp"


Eigen::VectorXd getExpectedVector(int a)
{
    Eigen::VectorXd res(10);
    for(int i=0; i<10; i++)
    {
        if(i == a)
            res[i] = 1;
        else
            res[i] = 0;
    }

    return res;
}

void a(std::vector<int> e)
{
    for(auto i: e)
        std::cout<<i << "\n";
}

int main(int argc, char *argv[])
{
    std::string trainFile = argv[1];
    Data trainingData;
    std::cout << "Reading training data..." << std::endl;
    trainingData.readFromFile(trainFile);

    std::string testFile = argv[2];
    Data testingData;
    std::cout << "Reading test data..." << std::endl;
    testingData.readFromFile(testFile);

    double lr = std::stod(argv[3]);

    NeuralNet nn({28*28, 15, 10});
    nn.compile();

    nn.train(trainingData, lr, 1);

    std::cout << "\nAccuracy: " << 100 * nn.eval(testingData)<< "%\n";

    return 0;
}