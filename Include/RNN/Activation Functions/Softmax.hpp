# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"


class Softmax : public RNNBaseClass
{   
public:
    Softmax()
    {
        type = ActivationFunction;
        name = "Softmax";
    }
    std::string GetName() const override
    {
        return name;
    }

private:
    
};