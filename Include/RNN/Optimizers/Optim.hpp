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
    Optim(OptimType _type = SGDOptim, double _lr = 0.001):type(_type),lr(_lr){}
    void step(numpy<numpy<double>> *weight, numpy<double> *bias, numpy<numpy<double>> weightChange, numpy<double> biasChange)
    {
        /*
            type: Decide which optimizer is used like Adam Optim, SGD Optim etc.
            => Each otpimizer class object is created and used according to what type of optimizer is called
            => learning rate and type are public variable so can be change any time in code
        */
        if(type == SGDOptim)
        {
            sgd.Update(weight, bias, weightChange, biasChange, lr);
        }

        else if(type == AdamOptim)
        {
            adm.Update(weight, bias, weightChange, biasChange, lr);
        }

        else if(type = AdaGradOptim)
        {
            
        }

        else
        {
            std::cout << "Not Exist" << std::endl;
        }
    }
private:
    SGD sgd;
    Adam adm;
};
