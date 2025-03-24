# pragma once

#include <iostream>
#include <string>


#include "numpy.hpp"
#include "BlockType.hpp"
#include "BaseClass.hpp"


class Softmax : public BaseClass
{   
public:
    Softmax()
    {
        type = ActivationFunction;
        name = "Softmax";
    }
    std::string GetName() const override
    {
        return name;
    }
    BlockType GetType() const override
    {
        return type;
    }

    numpy<double> Forward(numpy<double> input)
    {
        input.clip(-200, 200); // clip between -200 to 200
        input.exp(); // exp to all elements
        double total = input.sum(); // sum all the element of input
        for(int i = 0; i < input.size(); i++)
        {
            input[i] = input[i] / total;
        }
        saveProb = input;
        return input;
    }

    numpy<double> BackPropagate(numpy<double> prevGrad)
    {
        /*
            => [x1, x2, . . . , xn] is column vector
            Y = Softmax([x1, x2, . . . , xn])

            => Y = [exp(x1)/sigma(i, xi), . . . ., exp(xn)/sigma(i, xi)]

            => dy/dx = [
                            [d/dx1 (exp(x1)/sigma(i, xi)), . . . . ., d/dxn (exp(x1)/sigma(i, xi))],
                            [d/dx1 (exp(x2)/sigma(i, xi)), . . . . ., d/dxn (exp(x2)/sigma(i, xi))],
                            .
                            .
                            .
                            [d/dx1 (exp(xn)/sigma(i, xi)), . . . . ., d/dxn (exp(xn)/sigma(i, xi))]
                        ]
            d/dxi (exp(xi)/sigma(j, exp(xj))) => p[i](1 - p[i]) => -p[i] * (p[i] - 1)
            d/dxi (exp(xj)/sigma(j, exp(xj))) => -p[i] * p[j]
        */
       
        numpy<double> grad(saveProb.size(), 0);
        for(int i = 0; i < prevGrad.size(); i++)
        {
            if(prevGrad[i] == 0) // if prevGrad[i] = 0 then whole row will of softmax grad will be zero
                continue;
            numpy<double> copyProb(saveProb); // copy forward pass probability
            double pi = -copyProb[i]; // save p[i] element
            copyProb[i] -= 1; // make p[i] = p[i] - 1 and then multiply each element by -p[i]
            // can use directly multiply double var with numpy var. but there is one issue, it will also change same numpy var
            for(int j = 0; j < saveProb.size(); j++) 
            {
                copyProb[j] *= pi;
                grad[j] += (prevGrad[i] * copyProb[j]);
            }
        }
        return grad;
    }

private:
    numpy<double> saveProb;
};