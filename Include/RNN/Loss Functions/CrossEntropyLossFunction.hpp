# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"


class CrossEntropy : public RNNBaseClass
{   
public:
    CrossEntropy()
    {
        type = LossFunction;
        name = "CrossEntropy";
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
};