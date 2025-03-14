# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"

class Relu : public RNNBaseClass
{   
public:
    Relu()
    {
        name = "Relu";
        type = ActivationFunction;
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
        for(int i = 0; i < input.size(); i++)
        {
            if (input[i] < 0)
                input[i] = 0;
        }
        return input;
    }

private:
    
};