#include "Neuralnet.hpp"
#include "utils.hpp"

NeuralNet::NeuralNet(const std::vector<uint32_t> &sizes)
{
    _layers.reserve(sizes.size());
    _layers.emplace_back(sizes[0], -1, LayerType::input);
    for (int i = 1; i <= sizes.size() - 2; i++)
    {
        _layers.emplace_back(sizes[i], _layers[i - 1].getNeuronsCount(), LayerType::hidden);
    }
    _layers.emplace_back(sizes[sizes.size() - 1], _layers[sizes.size() - 2].getNeuronsCount(), LayerType::output);
}

void NeuralNet::compile()
{
    for (int i = 0; i < _layers.size() - 1; i++)
    {
        _layers[i].connect(&_layers[i + 1]);
    }
}

void NeuralNet::train(const Data &trainingData, double learningRate, uint32_t epochs)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "Epoch " << epoch << " of " << epochs << " started." << std::endl;
        for (int i = 0; i < trainingData.images.size(); i++)
        {
            _layers.front().feedInput(const_cast<Eigen::VectorXd &>(trainingData.images[i]));
            _layers.back().setExpectedOutput(digitVector(trainingData.expectedDigit[i]));

            for (auto &layer : _layers)
            {
                layer.feedForward();
            }

            for (auto it = _layers.rbegin(); it != _layers.rend(); it++)
            {
                it->backpropagate(learningRate);
            }

            // std::cout << _layers.back().getMeanSquareError() << std::endl;
        }
    }
}

double NeuralNet::eval(const Data &evalData, std::vector<std::pair<u_int32_t, u_int32_t>> &digitAccuracy)
{
    uint32_t correct = 0;

    for (int i = 0; i < evalData.images.size(); i++)
    {
        _layers.front().feedInput(const_cast<Eigen::VectorXd &>(evalData.images[i]));
        _layers.back().setExpectedOutput(digitVector(evalData.expectedDigit[i]));

        for (auto &layer : _layers)
        {
            layer.feedForward();
        }

        if (_layers.back().getPrediction() == evalData.expectedDigit[i])
        {
            correct += 1;
            ++digitAccuracy.at(_layers.back().getPrediction()).first;
        }
        else
        {
            ++digitAccuracy.at(_layers.back().getPrediction()).second;
        }
    }

    return ((double)correct) / ((double)evalData.images.size());
}

uint32_t NeuralNet::predict(const Eigen::VectorXd &input)
{
    _layers.front().feedInput(const_cast<Eigen::VectorXd &>(input));

    for (auto layer : _layers)
    {
        layer.feedForward();
    }

    return _layers.back().getPrediction();
}