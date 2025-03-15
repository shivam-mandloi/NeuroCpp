#include <iostream>
#include <memory>

#include "numpy.hpp"
#include "RNN.hpp"

int main()
{
    numpy<double> input = {1.2, 4.5, 0.212, 12.23, 123.1};
    Neuron nn1(input.size(), 10);
    Relu rl;
    Neuron nn2(10, 5);
    Softmax sf;
    std::cout << "output: ";
    std::cout << sf.Forward(nn2.Forward(rl.Forward(nn1.Forward(input)))) << std::endl;
    return 0;
}