#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <Eigen/Dense>
#include <vector>

enum class LayerType {input, hidden, output};

class Layer
{
public:
    Layer(size_t neurons, size_t inputs, LayerType layerType);
    
    void connect(Layer* layer);
    void feedForward();
    size_t getInputsCount();
    size_t getNeuronsCount();
    std::vector<double>& getDeltas();
    void feedInput(const Eigen::VectorXd& input);


private:
    Layer* _nextLayer;
    Eigen::VectorXd _inputs;
    Eigen::MatrixXd _weights;
    Eigen::VectorXd _outputs;
    Eigen::VectorXd _thresholds;
    size_t _neuronsCount = -1;
    size_t _inputsCount = -1;
    std::vector<double> _deltas;
    LayerType _layerType;

};



#endif