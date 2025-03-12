# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"


class Adam : public RNNBaseClass
{   
public:
    Adam()
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

    void Forward()
    {
        std::cout << "hello from: " << name << std::endl;
    }
private:
    BlockType type = Optimizer;
    std::string name = "Adam";
};