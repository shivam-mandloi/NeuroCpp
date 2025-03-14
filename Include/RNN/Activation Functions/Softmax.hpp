# pragma once

#include <iostream>
#include <string>


#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"


class Softmax : public RNNBaseClass
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
        return input;
    }

    private:
    double e = 2.718281828435;
};