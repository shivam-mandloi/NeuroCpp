# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"


class Neuron : public RNNBaseClass
{   
public:
    Neuron()
    {
        type = NeuronBlock;
        name = "Neuron";
    }

    std::string GetName() const override
    {
        return name;
    }

    BlockType GetType() const override
    {
        return type;
    }
    
    void Forward()
    {
        std::cout << weight << std::endl;
        std::cout << bias << std::endl;
    }
private:
    numpy<numpy<double>> weight = rand(2,3);
    numpy<double> bias = rand(3);
};