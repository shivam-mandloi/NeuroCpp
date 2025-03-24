# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "BaseClass.hpp"


class CrossEntropy : public BaseClass
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

    double Forward(numpy<double> predicted, numpy<double> actual)
    {
        double loss = 0;
        for(int i = 0; i < predicted.size(); i++)
        {
            loss += -(actual[i] * log(predicted[i]));
        }
        return loss;
    }

    numpy<double> BackPropagate(numpy<double> predicted, numpy<double> actual)
    {
        /*
            y = CrossEntropy(x)
            
            where:
                x' = [x'1, x'2, . . . . ., x'n] column vector
                x = [x1, x2, . . . ., xn], column vector.
                    x' => true distribution
                    x => predicted distribution
                
            y = -x'1 * log(x1) -x'2 * log(x2) - . . . . -x'n * log(xn)

            dy/dx = [-x'1 / x1, -x'2 / x2, . . . . , -x'n / xn]
        
        */
        numpy<double> grad(predicted.size(), 0); // initialize with zero
        for(int i = 0; i < predicted.size(); i++)
        {
            grad[i] = -actual[i] / predicted[i];
        }
        return grad;
    }
};