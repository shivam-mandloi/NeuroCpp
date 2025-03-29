# pragma once

#include <iostream>
#include <string>
#include <cmath>
#include "numpy.hpp"
#include "BlockType.hpp"


class Adam : public BaseClass
{   
public:
    Adam()
    {
        mtWeight = numpy<numpy<double>>();
        mtBias = numpy<double>();
        vtWeight = numpy<numpy<double>>();
        vtBias = numpy<double>();
        b1 = 0.9;
        b2 = 0.999;
        timeStep = 1;
        eps = 1e-10;
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
        /*
            W(t) = W(t-1) - n * (m(t) / ((V(t) ** 0.5) + eps))  | All matrix - matrix operation are element wise
            where,
                m(t) = m'(t) / (1 - b1**t)
                V(t) = v'(t) / (1 - b2**t)

                m'(t) = b1 * m'(t-1) + (1 - b1) * g(t) 
                v'(t) = b2 * v'(t-1) + (1 - b2) * (g(t) ** 2) | g(t) ** 2 is element wise multiplication between g(t)

                g(t) = t time derivative of loss wrt parameter
    
            => Best Initialization: m(0) = 0, v(0) = 0, b1 = 0.9, b2 = 0.999
        */

        // Update both weight and bias

        if(!mtWeight.size()) // Size is equal to zero, then intialize the mtWeight, vtWeight, mtBias, vtBias with zero
        {
            mtWeight = numpy<numpy<double>>(weightChange.size(), numpy<double>(weightChange[0].size(), 0));
            mtBias = numpy<double>(biasChange.size(), 0);
            vtWeight = numpy<numpy<double>>(weightChange.size(), numpy<double>(weightChange[0].size(), 0));
            vtBias = numpy<double>(biasChange.size(), 0);
        }

        // m'(t) = b1 * m'(t-1) + (1 - b1) * g(t) 
        for(int i = 0; i < mtWeight.size(); i++)
        {
            for(int j = 0; j < mtWeight[i].size(); j++)
            {
                mtWeight[i][j] = b1 * mtWeight[i][j] + (1-b1) * weightChange[i][j];
            }
            mtBias[i] = b1 * mtBias[i] + (1-b1) * biasChange[i];
        }

        // v'(t) = b2 * v'(t-1) + (1 - b2) * (g(t) ** 2)
        for(int i = 0; i < vtWeight.size(); i++)
        {
            for(int j = 0; j < vtWeight[i].size(); j++)
            {
                vtWeight[i][j] = b2 * vtWeight[i][j] + (1-b2) * (pow(weightChange[i][j], 2));
            }
            vtBias[i] = b2 * vtBias[i] + (1-b2) * pow(biasChange[i], 2);
        }

        // m(t) = m'(t) / (1 - b1**t)
        // V(t) = v'(t) / (1 - b2**t)
        // W(t) = W(t-1) - n * (m(t) / ((V(t) ** 0.5) + eps))
        // Combine above three operations
        for(int i = 0; i < vtWeight.size(); i++)
        {
            for(int j = 0; j < vtWeight[i].size(); j++)
            {
                double mtW = (mtWeight[i][j] / (1 - pow(b1, timeStep)));
                double vtW = (pow(vtWeight[i][j] / (1 - pow(b2, timeStep)), 0.5)) + eps;
                (*weight)[i][j] -= lr * (mtW / vtW);
            }
            double mtB = (mtBias[i] / (1 - pow(b1, timeStep)));
            double vtB = (pow(vtBias[i] / (1 - pow(b2, timeStep)), 0.5)) + eps;
            (*bias)[i] -= lr * (mtB / vtB);
        }
        timeStep+=1;
        std::cout << "done " << timeStep << std::endl;
    }
private:
    BlockType type = Optimizer;
    std::string name = "Adam";
    numpy<numpy<double>> mtWeight;
    numpy<numpy<double>> vtWeight;
    numpy<double> mtBias;
    numpy<double> vtBias;
    double b1;
    double b2;
    int timeStep;
    double eps;
};