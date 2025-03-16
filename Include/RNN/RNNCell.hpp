# pragma once

#include <iostream>
#include <string>
#include <memory>

#include "numpy.hpp"
#include "BlockType.hpp"

#include "RNNBaseClass.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "CrossEntropyLossFunction.hpp"
#include "Adam.hpp"
#include "SGD.hpp"
#include "Neuron.hpp"


class RNNCell
{
public:
    RNNCell(int EmbeddingSize, int VocabSize, int hiddenLayerSize, int OuputSize = -1)
    : E(EmbeddingSize, VocabSize), V(hiddenLayerSize, hiddenLayerSize), U(hiddenLayerSize, hiddenLayerSize)
    {
        
    }

    void Forward()
    {
    }

private:
    Neuron E;
    Neuron V;
    Neuron U;
    Relu reluFunc;
    Softmax sfFunc;
    CrossEntropy lossFunc;
    Adam admOpt;
};

