# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"

class Relu : public RNNBaseClass
{   
public:
    Relu()
    {
        name = "Relu";
        type = ActivationFunction;
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
        saveInput = input;
        for(int i = 0; i < input.size(); i++)
        {
            if (input[i] < 0)
                input[i] = 0;
        }
        return input;
    }

    numpy<double> BackPropagate(numpy<double> prevGrad)
    {
        /*
            => Finding gradient of Relu function give us matrix.
            => It will be diagonal matrix.

                y = [y1,
                     y2,
                    -y3,
                    -y4,
                     y5]
            => Suppose y is column vector.
            
            then Relu(y) = [y1, y2,0, 0, y5]

            => When we have to BackPropagat, it become

                d Relu(y)/dy = Matrix
        */
        for(int i = 0; i < saveInput.size(); i++)
        {
            if(saveInput[i] <= 0)
                prevGrad[i] = 0;
        }
        return prevGrad;
    }

private:
    numpy<double> saveInput;
    
};