# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"


class SGD : public RNNBaseClass
{   
public:
    SGD()
    {
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
    BlockType type = Optimizer;
    std::string name = "SGD";
    
};