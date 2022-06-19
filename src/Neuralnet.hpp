#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <stdint.h>
#include "Data.hpp"
#include <Eigen/Dense>
#include "Layer.hpp"


class NeuralNet
{
public:

    NeuralNet(const std::vector<uint32_t>& sizes);
    void compile();
    void train(const Data& trainingData, double learningRate, uint32_t epochs);
    double eval(const Data& evalData);
    uint32_t predict(const Eigen::VectorXd& input);

private:

    std::vector<Layer> _layers;


};

#endif