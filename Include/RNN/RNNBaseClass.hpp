#pragma once

#include <string>
#include "BlockType.hpp"

/*
    Inherit by all the block class in RNN, so that we can create vector of class pointer
*/


class RNNBaseClass
{
public:
    RNNBaseClass(){}
    virtual std::string GetName() const = 0;
    virtual BlockType GetType() const = 0;
protected:
    BlockType type;
    std::string name;
};