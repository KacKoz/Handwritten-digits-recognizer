#include "Data.hpp"
#include <fstream>
#include <iostream>
#include "stdio.h"
#include <iostream>
#include "Neuralnet.hpp"
#include "utils.hpp"


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
    int epochs = std::stoi(argv[4]);
    //Layer input(28*28, 0, LayerType::input);


    Layer input(28*28, 1, LayerType::input);
    Layer hidden(15, 28*28, LayerType::hidden);
    Layer output(10, 15, LayerType::output);

    input.connect(&hidden);
    hidden.connect(&output);

    for(int epoch=0; epoch<epochs; epoch++)
    {
        std::cout << "Starting epoch " << epoch+1 << " of " << epochs << std::endl;
        for(int i=0; i<trainingData.images.size(); i++)
        {
            input.feedInput(trainingData.images[i]);
            output.setExpectedOutput(digitVector(trainingData.expectedDigit[i]));

            input.feedForward();
            hidden.feedForward();
            output.feedForward();

            output.backpropagate(lr);
            hidden.backpropagate(lr);
            input.backpropagate(lr);

            //std::cout << output.getMeanSquareError() << std::endl;

        }
    }


    uint32_t correct = 0;
    for(int i=0; i<testingData.images.size(); i++)
    {
        input.feedInput(testingData.images[i]);
        //output.setExpectedOutput(digitVector(testingData.expectedDigit[i]));

        input.feedForward();
        hidden.feedForward();
        output.feedForward();

        if(output.getPrediction() == testingData.expectedDigit[i])
        {
            correct++;
        }
    }

    std::cout << "Accuracy: " << 100 * ((double)correct)/((double)testingData.images.size()) << "%" << std::endl;




    // NeuralNet nn({28*28, 15, 10});
    // nn.compile();

    // nn.train(trainingData, lr, 1);

    // std::cout << "\nAccuracy: " << 100 * nn.eval(testingData)<< "%\n";

    return 0;
}