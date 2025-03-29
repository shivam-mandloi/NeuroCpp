# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "BaseClass.hpp"
#include "NumpyHelpingFunc.hpp"
#include "Optim.hpp"

/*
    will update each matrix-matrix, vector-vector, matrix-scalar and vector-scalar operation with seperate function in numpy class
*/

class Neuron : public BaseClass
{   
public:
    Neuron(int inputSize, int OutputSize, OptimType _optType = SGDOptim, double _lr = 0.001, int _batchSize = 1)
    {
        type = NeuronBlock;
        name = "Neuron";
        weight = hf.rand(OutputSize, inputSize);
        bias = hf.rand(OutputSize);
        opt.type = _optType; // change default optimization type
        opt.lr = _lr; // change default leraning rate 
        avgWeightChange = numpy<numpy<double>>(weight.size(), numpy<double>(weight[0].size(), 0)); // same size as weight matrix
        avgBiasChange = numpy<double>(bias.size(), 0);
        currentBatchInput = 1;
        batchSize = _batchSize;
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
        return hf.MatMul(weight, input) + bias;
    }

    numpy<double> BackPropagate(numpy<double> prevGrad)
    {
        /*
            y = Wx + b
            
            where:
                w = [[w11, w12, . . . ., w1n],
                    [w21, w22, . . . ., w2n],
                    .
                    .
                    .
                    .
                    [wm1, wm2, . . . ., wmn]]

                x = [x1, x2, . . . ., xn] column vector

                b = [b1, b2, . . . ., bm] column vector
            
            y = [w11*x1 + w12*x2 + . . . . + w1n*xn + b1,
                w21*x1 + w22*x2 + . . . . + w2n*xn + b2,
                .
                .
                .
                .
                wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm
                ]
            
            dy/dw = [
                        [
                            [d/dw11 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), . . . . ., d/dw1n (w11*x1 + w12*x2 + . . . . + w1n*xn + b1)],
                            [d/dw21 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), . . . . ., d/dw2n (w11*x1 + w12*x2 + . . . . + w1n*xn + b1)],
                            .
                            .
                            .
                            [d/dwm1 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), . . . . ., d/dwmn (w11*x1 + w12*x2 + . . . . + w1n*xn + b1)]
                        ],

                        [
                            [d/dw11 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), . . . . ., d/dw1n (w21*x1 + w22*x2 + . . . . + w2n*xn + b2)],
                            [d/dw21 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), . . . . ., d/dw2n (w21*x1 + w22*x2 + . . . . + w2n*xn + b2)],
                            .
                            .
                            .
                            [d/dwm1 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), . . . . ., d/dwmn (w21*x1 + w22*x2 + . . . . + w2n*xn + b2)]
                        ],

                        .
                        .
                        .
                        .
                        .,

                        [
                            [d/dw11 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), . . . . ., d/dw1n (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm)],
                            [d/dw21 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), . . . . ., d/dw2n (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm)],
                            .
                            .
                            .
                            [d/dwm1 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), . . . . ., d/dwmn (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm)]
                        ],
                    ]
            => Here for first matrix, only first row will be non zero, for second matrix second row will be non zero and so on

            prevGrad = [p1, p2, . . . ., pm] row vector

            changeInWeightMatrix (W) => prevGrad @ dy/dw
                                       
                                           [p1,
                                            p2,   
                                            .
            changeInWeightMatrix (W) =>     .     @   [x1, x2, . . . ., xn]
                                            . 
                                            .,
                                            pm]

            changeInBias (b) => prevGrad @ I
                                I => Identity matrix of size m X m

            nextGrad => grad which is passed to previous layer in chain rule of Back Propagation

            nextGrad => prevGrad @ dy/dx

            dy/dx = [
                        [d/dx1 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), d/dx2 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), . . . ., d/dxn (w11*x1 + w12*x2 + . . . . + w1n*xn + b1) ],
                        [d/dx1 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), d/dx2 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), . . . ., d/dxn (w21*x1 + w22*x2 + . . . . + w2n*xn + b2)],
                        .
                        .
                        .
                        .
                        [d/dx1 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), d/dx2 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), . . . ., d/dxn (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm)]
                    ]

            dy/dx = [[w11, w12, . . . ., w1n],
                    [w21, w22, . . . ., w2n],
                    .
                    .
                    .
                    .
                    [wm1, wm2, . . . ., wmn]]
        */    
        numpy<numpy<double>> changeInWeight(prevGrad.size(), numpy<double>(saveInput.size(), 0.0));

        for(int i = 0; i < prevGrad.size(); i++)
        {
            for(int j = 0; j < saveInput.size(); j++)
            {
                changeInWeight[i][j] = prevGrad[i] * saveInput[j]; // find the grad of loss wrt weights
            }
        }
        
        numpy<double> nextGrad(saveInput.size(), 0); 
        for(int i = 0; i < saveInput.size(); i++) 
        {
            double temp = 0.0;
            for(int j = 0; j < prevGrad.size(); j++)
            {
                temp += (prevGrad[j] * weight[j][i]);// grad for next layer
            }
            nextGrad[i] = temp;
        }

        for (int i = 0; i < weight.size(); i++) // Add the parameter to avgWeight and avgBias
        {
            for(int j = 0; j < weight[i].size(); j++)
            {
                avgWeightChange[i][j] = avgWeightChange[i][j] + changeInWeight[i][j];

            }
            avgBiasChange[i] = avgBiasChange[i] + prevGrad[i];
        }
        
        if(currentBatchInput == batchSize) // update the weights
        {
            std::cout << currentBatchInput << " " << batchSize << std::endl;
            for (int i = 0; i < weight.size(); i++) // average the weight matrix and bias
            {
                for(int j = 0; j < weight[i].size(); j++)
                {
                    avgWeightChange[i][j] = avgWeightChange[i][j] / batchSize;

                }
                avgBiasChange[i] = avgBiasChange[i] / batchSize;
            }

            opt.step(&weight, &bias, avgWeightChange, avgBiasChange); // update change according to optimizer
            currentBatchInput = 0;
            avgWeightChange = numpy<numpy<double>>(weight.size(), numpy<double>(weight[0].size(), 0));
            avgBiasChange = numpy<double>(bias.size(), 0);
        }
        currentBatchInput++;
        return nextGrad;
    }

private:
    numpy<numpy<double>> weight;
    numpy<double> bias;
    NpHelpingFunc hf;
    numpy<double> saveInput;
    Optim opt;
    int currentBatchInput;
    numpy<numpy<double>> avgWeightChange;
    numpy<double> avgBiasChange;
    int batchSize;
};