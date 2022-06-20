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

    NeuralNet(const std::vector<size_t>& sizes);
    NeuralNet(){};
    void compile();
    void train(const Data& trainingData, double learningRate, uint32_t epochs);
    double eval(const Data &evalData, std::vector<std::pair<u_int32_t, u_int32_t>> &digitAccuracy);
    uint32_t predict(const Eigen::VectorXd& input);
    void load(std::string filename);
    void save(std::string filename);

private:
    void _createLayers(const std::vector<size_t>& sizes);
    void _feedThrough(Eigen::VectorXd& image, int expectedOutput);
    void _feedThrough(Eigen::VectorXd& image);
    void _backpropagate(double learningRate);

    std::vector<Layer> _layers;
};

#endif