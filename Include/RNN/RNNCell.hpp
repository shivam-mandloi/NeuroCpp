# pragma once

#include <iostream>
#include <string>
#include <memory>

#include "numpy.hpp"
#include "BlockType.hpp"

#include "RNNBaseClass.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "CrossEntropyLossFunction.hpp"
#include "Adam.hpp"
#include "SGD.hpp"
#include "Neuron.hpp"


class RNN
{   
public:
    RNN()
    {

    }
    
    void AddBlock(std::unique_ptr<RNNBaseClass> block)
    {
        blocks.push_back(std::move(block));
    }

    void PrintALLBlock()
    {
        for(int i = 0; i < blocks.size(); i++)
        {
            std::cout << blocks[i]->GetName() << " ";
        }
        std::cout << std::endl;
    }

private:
    numpy<std::unique_ptr<RNNBaseClass>> blocks = {};
};