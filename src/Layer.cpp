#include <Eigen/Dense>
#include <random>
#include "Layer.hpp"
#include "utils.hpp"

Layer::Layer(size_t neurons, size_t inputs, LayerType layerType) 
    : _layerType(layerType)
{
    _outputs = Eigen::VectorXd::Zero(neurons + ((layerType == LayerType::output) ? 0 : 1));
    _inputs = Eigen::VectorXd::Zero(neurons + ((layerType == LayerType::input) ? 0 : 1));

    _neuronsCount = _outputs.size();
    _inputsCount = _inputs.size();

    _deltas.resize(_neuronsCount);
}

void Layer::connect(Layer* layer)
{
    _nextLayer = layer;

    double stddev_weights = 2.38/sqrt(static_cast<double>(_inputsCount));

    std::default_random_engine generator;
    std::normal_distribution<double> distribution_weights(0.0,stddev_weights);

    _weights = Eigen::MatrixXd::Zero(_neuronsCount, layer->getNeuronsCount());
    for(size_t i=0; i < _neuronsCount; i++)
    {
        for(size_t j=0; j<layer->getNeuronsCount(); j++)
        {
            _weights(i, j) = distribution_weights(generator);
        }
    }
}

void Layer::feedInput(const Eigen::VectorXd& input)
{
    if(input.size() != _inputs.size())
        throw std::runtime_error("Wrong input vector size!");

    for(size_t i=0; i < _inputs.size(); i++)
    {
        _inputs[i] = input[i];
    }
}

void Layer::feedForward()
{
    if(_layerType == LayerType::input)
    {
        for(size_t i=0; i < _neuronsCount; i++)
        {
            _outputs[i] = _inputs[i];
        }     
    }
    else
    {
        for(size_t i=0; i < _neuronsCount; i++)
        {
            _outputs[i] = sigmoid(_inputs[i]);
        }
    }


    if(_layerType != LayerType::output)
    {
        Eigen::VectorXd nextLayerInput = _weights*_outputs;
        _nextLayer->feedInput(nextLayerInput);
    }
}

std::vector<double>& Layer::getDeltas()
{
    return _deltas;
}

size_t Layer::getNeuronsCount()
{
    return _neuronsCount;
}