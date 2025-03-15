# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"
#include "NumpyHelpingFunc.hpp"


class Neuron : public RNNBaseClass
{   
public:
    Neuron(int inputSize, int OutputSize)
    {
        type = NeuronBlock;
        name = "Neuron";
        weight = rand(OutputSize, inputSize);
        bias = rand(OutputSize);
    }

    std::string GetName() const override
    {
        return name;
    }

    BlockType GetType() const override
    {
        return type;
    }
    
    numpy<double> Forward(numpy<double> input)
    {
        return hf.MatMul(weight, input) + bias;
    }
private:
    numpy<numpy<double>> weight;
    numpy<double> bias;
    NpHelpingFunc hf;
};