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
private:
};