#include "Neuralnet.hpp"
#include "utils.hpp"


NeuralNet::NeuralNet(const std::vector<size_t>& sizes)
{
    _createLayers(sizes);
}

void NeuralNet::_createLayers(const std::vector<size_t>& sizes)
{
    _layers.reserve(sizes.size());
    _layers.emplace_back(sizes[0], -1, LayerType::input);
    for(int i=1; i<=sizes.size()-2; i++)
    {
        _layers.emplace_back(sizes[i], _layers[i-1].getNeuronsCount()+1, LayerType::hidden);
    }
    _layers.emplace_back(sizes[sizes.size()-1], _layers[sizes.size()-2].getNeuronsCount(), LayerType::output);
}

void NeuralNet::compile()
{
    for(int i=0; i<_layers.size()-1; i++)
    {
        _layers[i].connect(&_layers[i+1]);
    }
}

void NeuralNet::_backpropagate(double learningRate)
{
    for(int i=_layers.size()-1; i>=0; i--)
    {
        _layers[i].backpropagate(learningRate);
    }
}

void NeuralNet::train(const Data& trainingData, double learningRate, uint32_t epochs)
{
    for(int epoch=0; epoch<epochs; epoch++)
    {
        std::cout << "Epoch " << epoch+1 << " of " << epochs << " started." << std::endl;
        for(int i=0; i<trainingData.images.size(); i++)
        {
            _feedThrough(const_cast<Eigen::VectorXd&>(trainingData.images[i]), trainingData.expectedDigit[i]);
            _backpropagate(learningRate);
        }
    }
}

void NeuralNet::_feedThrough(Eigen::VectorXd& image, int expectedOutput)
{
    _layers.front().feedInput(const_cast<Eigen::VectorXd&>(image));
    _layers.back().setExpectedOutput(digitVector(expectedOutput));

    for(auto& layer: _layers)
    {
        layer.feedForward();
    }
}

double NeuralNet::eval(const Data& evalData)
{
    uint32_t correct = 0;

    for(int i=0; i<evalData.images.size(); i++)
    {
        _feedThrough(const_cast<Eigen::VectorXd&>(evalData.images[i]), evalData.expectedDigit[i]);

        if(_layers.back().getPrediction() == evalData.expectedDigit[i])
        {
            correct += 1;
        }
    }

    return ((double)correct)/((double)evalData.images.size());
}

uint32_t NeuralNet::predict(const Eigen::VectorXd& input)
{
    _layers.front().feedInput(const_cast<Eigen::VectorXd&>(input));

    for(auto layer: _layers)
    {
        layer.feedForward();
    }

    return _layers.back().getPrediction();
}

void NeuralNet::save(std::string filename)
{
    std::ofstream ofs(filename, std::ios::out);
    ofs << _layers.size() << std::endl;
    for(auto& layer: _layers)
    {
        ofs << layer.getNeuronsCount() << std::endl;
    }

    for(auto& layer: _layers)
    {
        layer.writeWeightsData(ofs);
    }

    ofs.close();
}

void NeuralNet::load(std::string filename)
{
    std::ifstream ifs(filename, std::ios::in);
    size_t layers;
    ifs >> layers;
    std::vector<size_t> sizes;
    for(int i=0; i<layers; i++)
    {
        size_t tmp;
        ifs >> tmp;
        sizes.push_back(tmp);
    }

    _createLayers(sizes);
    compile();

    for(auto& layer: _layers)
    {   
        layer.readWeightsData(ifs);
    }

    ifs.close();
}