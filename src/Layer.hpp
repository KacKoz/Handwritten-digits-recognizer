#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <Eigen/Dense>
#include <vector>

enum class LayerType {input, hidden, output};

class Layer
{
public:
    Layer(size_t neurons, size_t previousLayerOutputs, LayerType layerType);
    
    void connect(Layer* layer);
    void feedForward();
    size_t getInputsCount();
    size_t getNeuronsCount();
    const Eigen::VectorXd& getDeltas();
    void setExpectedOutput(const Eigen::VectorXd& expectedOutput);
    void feedInput(Eigen::VectorXd& input);
    void backpropagate(double learningRate);
    double getMeanSquareError();
    int getPrediction();

private:
    void _calculateDeltas();
    void _calculateOutputDeltas();
    void _calculateNonOutputDeltas();
    void _updateWeights(double learningRate);

    Layer* _nextLayer;
    Eigen::VectorXd _inputs;
    Eigen::MatrixXd _weights;
    Eigen::VectorXd _outputs;
    Eigen::VectorXd _thresholds;
    LayerType _layerType;
    size_t _neuronsCount;
    size_t _previousLayerOutputs;
    Eigen::VectorXd _deltas;
    Eigen::VectorXd _expectedOutput;
    double _learningRate;

};



#endif