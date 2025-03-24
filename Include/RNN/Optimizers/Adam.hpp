# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"


class Adam : public BaseClass
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

    void Update(numpy<numpy<double>> *weight, numpy<double> *bias, numpy<numpy<double>> weightChange, numpy<double> *biasChange)
    {
        
    }
private:
    BlockType type = Optimizer;
    std::string name = "Adam";
};