#include <iostream>
#include <memory>

#include "numpy.hpp"
#include "RNN.hpp"
#include "NumpyHelpingFunc.hpp"

struct nn
{
    Neuron nn1;
    Relu rl1;
    Neuron nn2;
    Relu rl2;
    Neuron nn3;
    Softmax sf;
    CrossEntropy ls;
    nn() : nn1(4, 8), nn2(8, 6), nn3(6, 3){}

    numpy<double> Forward(numpy<double> input)
    {
        return sf.Forward(nn3.Forward(rl2.Forward(nn2.Forward(rl1.Forward(nn1.Forward(input))))));
    }

    void BackPropagate(numpy<double> pred, numpy<double> actual)
    {
        nn1.BackPropagate(rl1.BackPropagate(nn2.BackPropagate(rl2.BackPropagate(nn3.BackPropagate(sf.BackPropagate(ls.BackPropagate(pred, actual)))))));
    }

    double loss(numpy<double> input, int index)
    {
        numpy<double> actual(input.size(), 0);
        actual[index] = 1;
        return ls.Forward(input, actual);
    }
};


int main()
{
    NpHelpingFunc hf;

    numpy<double> dataStr = hf.SplitString(hf.ReadFile("C:\\Users\\shiva\\Desktop\\IISC\\temp\\New folder (2)\\Data\\Iris.txt"));
    numpy<numpy<double>> data;
    numpy<double> temp = {};

    for(int i = 0; i < dataStr.size(); i++)
    {
        temp.push_back(dataStr[i]);
        if(temp.size() == 5)
        {
            data.push_back(temp);
            temp = {};
        }
    }

    nn network;
    for(int i = 0; i < data.size(); i++)
    {
        numpy<double> input = {data[i][0], data[i][1], data[i][2], data[i][3]};
        int target = data[i][4];
        double ls = network.loss(network.Forward(input), target);
        std::cout << ls << std::endl;
    }
    return 0;
}