#include "Data.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include "stdio.h"
#include <iostream>
#include "Neuralnet.hpp"
#include "utils.hpp"

#include "csvfile.h"


int main(int argc, char *argv[])
{
    // std::string trainFile = argv[1];
    // Data trainingData;
    // std::cout << "Reading training data..." << std::endl;
    // trainingData.readFromFile(trainFile);

    // std::string testFile = argv[2];
    // Data testingData;
    // std::cout << "Reading test data..." << std::endl;
    // testingData.readFromFile(testFile);

    // double lr = std::stod(argv[3]);
    // int epochs = std::stoi(argv[4]);

    // NeuralNet nn({28*28, 25, 10});
    // nn.compile();
    // nn.train(trainingData, lr, epochs);


    // nn.save("model");

    NeuralNet nn;
    nn.load("model");
    Eigen::VectorXd input(28*28);
    // if(argc != (1 + (28*28)))
    //     return -1;

    std::ifstream ifs;
    ifs.open(argv[1]);

    for(int i=1; i<=28*28; i++)
    {
        int tmp;
        ifs >> tmp;
        input[i-1] = tmp;
    }

    ifs.close();


    int prediction = nn.predict(input);
    std::cout << prediction << std::endl;

    //std::cout << "\nAccuracy: " << 100 * nn.eval(testingData) << "%\n";



    //std::string outputFilename = argv[5];

    // std::vector<std::pair<u_int32_t, uint32_t>> digitAccuracyVec = accuracyVector();


    // double accuracy = nn.eval(testingData, digitAccuracyVec);

    // std::cout << "\nAccuracy: " << 100 * accuracy << "%\n";

    // try
    // {
    //     csvfile csv(outputFilename);
    //     writeResults(csv, epochs, lr, accuracy, digitAccuracyVec);
    // }
    // catch (const std::exception &ex)
    // {
    //     std::cout << "Exception was thrown: " << ex.what() << std::endl;
    // }
  
    return prediction;
}