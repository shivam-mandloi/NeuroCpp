# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"


class Adam : public RNNBaseClass
{   
public:
    Adam()
    {
        type = Optimizer;
        name = "Adam";
    }

    std::string GetName() const override
    {
        return name;
    }

private:
    
};