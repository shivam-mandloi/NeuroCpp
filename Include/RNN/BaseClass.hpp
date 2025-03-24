#pragma once

#include <string>
#include "BlockType.hpp"


/*
    Inherit by all the block class in RNN. To have common parent
*/


class BaseClass
{
public:
    BaseClass(){}
    virtual std::string GetName() const = 0;
    virtual BlockType GetType() const = 0;
protected:
    BlockType type;
    std::string name;
};