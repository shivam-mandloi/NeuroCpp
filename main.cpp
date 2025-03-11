#include <iostream>
#include <memory>

#include "numpy.hpp"
#include "RNNCell.hpp"

int main()
{
    RNN rnn;
    rnn.AddBlock(std::make_unique<Neuron>());
    rnn.AddBlock(std::make_unique<Relu>());
    rnn.AddBlock(std::make_unique<Neuron>());
    rnn.AddBlock(std::make_unique<Softmax>());
    rnn.AddBlock(std::make_unique<CrossEntropy>());
    rnn.PrintALLBlock();
    return 0;
}