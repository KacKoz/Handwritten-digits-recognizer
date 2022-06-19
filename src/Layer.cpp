#include <Eigen/Dense>
#include <random>
#include "Layer.hpp"
#include "utils.hpp"
#include <iostream>


Layer::Layer(size_t neurons, size_t previousLayerOutputs, LayerType layerType) 
    : _layerType(layerType), _neuronsCount(neurons), _previousLayerOutputs(previousLayerOutputs)
{
    if(layerType == LayerType::input)
    {
        _previousLayerOutputs = 1;
    }

    _outputs = Eigen::VectorXd::Zero(neurons + ((layerType == LayerType::output) ? 0 : 1));
    _inputs = Eigen::VectorXd::Zero(neurons);

    _outputs[0] = 1;
    
    _deltas.resize(neurons);
}

void Layer::connect(Layer* layer)
{
    _nextLayer = layer;

    double stddev_weights = 2.38/sqrt(static_cast<double>(_previousLayerOutputs));

    std::default_random_engine generator;
    std::normal_distribution<double> distribution_weights(0.0,stddev_weights);

    _weights = Eigen::MatrixXd::Zero(layer->getNeuronsCount(), _neuronsCount + 1);
    for(size_t i=0; i < _weights.rows(); i++)
    {
        for(size_t j=0; j<_weights.cols(); j++)
        {
            _weights(i, j) = distribution_weights(generator);
        }
    }
}

void Layer::feedInput(Eigen::VectorXd& input)
{
    if(input.size() != _inputs.size())
        throw std::runtime_error("Wrong input vector size!");

    if(_layerType == LayerType::input)
    {
        for(size_t i=1; i < _neuronsCount; i++)
        {
            _outputs[i] = input[i];
        }
    }
    else
    {
        _inputs = std::move(input);
    }
}

void Layer::feedForward()
{
    if(_layerType != LayerType::input)
    {
        int shift = ((_layerType == LayerType::output) ? 0 : 1);
        for(size_t i=0; i < _inputs.size(); i++)
        {
            _outputs[i + shift] = sigmoid(_inputs[i]);
        }
    }

    if(_layerType != LayerType::output)
    {
        Eigen::VectorXd nextLayerInput = _weights*_outputs;
        _nextLayer->feedInput(nextLayerInput);
    }
}

void Layer::setExpectedOutput(const Eigen::VectorXd& expectedOutput)
{
    if(_layerType == LayerType::output)
    {
        _expectedOutput = expectedOutput;
    }
}

const Eigen::VectorXd& Layer::getDeltas()
{
    return _deltas;
}

size_t Layer::getNeuronsCount()
{
    return _neuronsCount;
}

void Layer::_calculateOutputDeltas()
{
    for(int i=0; i<_neuronsCount; i++)
    {
        _deltas[i] = (_expectedOutput[i]-_outputs[i]) * _outputs[i] * (1 - _outputs[i]);
    }
}

void Layer::_calculateNonOutputDeltas()
{
    auto neuronWeights = _weights.rightCols(_neuronsCount); // without bias (threshold) weights
    _deltas = neuronWeights.transpose() * _nextLayer->getDeltas();
    for(int i=0; i<_deltas.size(); i++)
    {
        _deltas[i] *= _outputs[i+1] * (1 - _outputs[i+1]);
    }
}

void Layer::_updateWeights(double learningRate)
{
    _weights += learningRate * (_nextLayer->getDeltas() * _outputs.transpose());
}

void Layer::_calculateDeltas()
{
    if(_layerType == LayerType::output)
    {
        _calculateOutputDeltas();
    }
    else
    {
        _calculateNonOutputDeltas();
    }
}

void Layer::backpropagate(double learningRate)
{
    _calculateDeltas();
    if(_layerType != LayerType::output)
    {
        _updateWeights(learningRate);
    }
}

double Layer::getMeanSquareError()
{
    if(_layerType != LayerType::output)
        throw std::runtime_error("MSE should only be calculated in output layer!");

    double error = 0;
    for(int i=0; i<_neuronsCount; i++)
    {
        error += 0.5 * pow(_expectedOutput[i] - _outputs[i], 2);
    }

    return error;
}

int Layer::getPrediction()
{
    int prediction = 0;

    for(int i=1; i<_neuronsCount; i++)
    {
        if(_outputs[i] > _outputs[prediction])
        {
            prediction = i;
        }
    }

    return prediction;
}