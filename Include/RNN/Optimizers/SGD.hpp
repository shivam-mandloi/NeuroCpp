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

    void Update(numpy<numpy<double>> *weight, numpy<double> *bias, numpy<numpy<double>> weightChange, numpy<double> biasChange, double lr = 0.001)
    {
        for(int i = 0; i < (*weight).size(); i++)
        {
            for(int j = 0; j < (*weight)[i].size(); j++)
            {
                (*weight)[i][j] -= (lr * weightChange[i][j]);
            }
            (*bias)[i] -= (lr * biasChange[i]); // change in bias is prevgrad @ Identity matrix, which make prevGrad to column matrix
        }
    }

private:
    BlockType type = Optimizer;
    std::string name = "SGD";
};