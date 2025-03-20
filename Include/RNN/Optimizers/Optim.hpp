#pragma once

#include "numpy.hpp"
#include "Adam.hpp"
#include "SGD.hpp"

enum OptimType
{
    AdamOptim,
    SGDOptim,
    AdaGradOptim
};


class Optim
{
public:
    OptimType type;
    double lr;
    Optim(OptimType _type = SGDOptim, double _lr = 0.01):type(_type),lr(_lr){}
    void step(numpy<numpy<double>> *weight, numpy<double> *bias, numpy<numpy<double>> weightChange, numpy<double> biasChange)
    {
        if(type == SGDOptim)
        {

        }

        else if(type == AdamOptim)
        {

        }

        else if(type = AdaGradOptim)
        {

        }
    }
private:
    SGD sgd;
    Adam adm;
};
