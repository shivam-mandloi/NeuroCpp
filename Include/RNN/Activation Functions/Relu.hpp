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

    void Forward()
    {
        std::cout << "forword from: " << name << std::endl;
    }

private:
    
};