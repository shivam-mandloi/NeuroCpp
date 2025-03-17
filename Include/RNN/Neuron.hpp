# pragma once

#include <iostream>
#include <string>

#include "numpy.hpp"
#include "BlockType.hpp"
#include "RNNBaseClass.hpp"
#include "NumpyHelpingFunc.hpp"


class Neuron : public RNNBaseClass
{   
public:
    Neuron(int inputSize, int OutputSize)
    {
        type = NeuronBlock;
        name = "Neuron";
        weight = rand(OutputSize, inputSize);
        bias = rand(OutputSize);
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

                b = [b1, b2, . . . ., bm] row vector
            
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
            changeInWeightMatrix (W) => [p1,
                                        p2,   
                                        .
                                        .      @  [x1, x2, . . . ., xn]
                                        . 
                                        .,
                                        pm]

            changeInBias (b) => prevGrad @ I
                                I => Identity matrix of size m X m

            nextGrad => grad which is passed to previous layer in chain rule of Back Propagation

            nextGrad => prevGrad @ dy/dx

            dy/dx = [[d/dx1 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), d/dx2 (w11*x1 + w12*x2 + . . . . + w1n*xn + b1), . . . ., d/dxn (w11*x1 + w12*x2 + . . . . + w1n*xn + b1) ],
                    [[d/dx1 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), d/dx2 (w21*x1 + w22*x2 + . . . . + w2n*xn + b2), . . . ., d/dxn (w21*x1 + w22*x2 + . . . . + w2n*xn + b2)],
                    .
                    .
                    .
                    .
                    [[d/dx1 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), d/dx2 (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm), . . . ., d/dxn (wm1*x1 + wm2*x2 + . . . . + wmn*xn + bm)]
                    ]

            dy/dx => [[w11, w12, . . . ., w1n],
                    [w21, w22, . . . ., w2n],
                    .
                    .
                    .
                    .
                    [wm1, wm2, . . . ., wmn]
                    ]
        */
       

    }

private:
    numpy<numpy<double>> weight;
    numpy<double> bias;
    NpHelpingFunc hf;
    numpy<double> saveInput;
};