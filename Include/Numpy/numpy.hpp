#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <math.h>


/*
Usage:
    numpy<LONG> np2 = {1,2,3};
    numpy<LONG> np1 = {2,3,4};
    print << (np1 * np2) << end;
    LONG var = 2;
    np1 =  np1 * var;
    np1 = np1 + np2;
    print << np1 << end;
    print << (np2 - np1) << end;
    print << (np1 * -1) << end;
    print << "Norm " << np1.norm() << end;
*/

template <typename T>
class numpy : public std::vector<T>
{
public:
    using std::vector<T>::vector;
    numpy operator*(const numpy &npy) const
    {
        if (npy.size() != this->size())
        {
            std::cerr << "size doesn't match" << std::endl;
            return numpy();
        }
        numpy res;
        for (int i = 0; i < npy.size(); i++)
        {
            res.push_back(npy[i] * (*this)[i]);
        }
        return res;
    }

    numpy operator*(const int var)
    {
        for (int i = 0; i < this->size(); i++)
        {
            (*this)[i] *= var;
        }
        return *this;
    }

    numpy operator*(const double var)
    {
        for (int i = 0; i < this->size(); i++)
        {
            (*this)[i] *= var;
        }
        return *this;
    }

    numpy operator+(const numpy npy)
    {
        if (npy.size() != this->size())
        {
            std::cerr << "size doesn't match" << std::endl;
            return numpy();
        }
        numpy resNp;
        for (int i = 0; i < npy.size(); i++)
        {
            resNp.push_back(npy[i] + (*this)[i]);
        }
        return resNp;
    }

    numpy operator+(long long val)
    {
        numpy resNp;
        for (int i = 0; i < this->size(); i++)
        {
            resNp.push_back(val + (*this)[i]);
        }
        return resNp;
    }

    numpy operator-(const numpy npy)
    {
        if (npy.size() != this->size())
        {
            std::cerr << "size doesn't match" << std::endl;
            return numpy();
        }
        numpy resNp;
        for (int i = 0; i < npy.size(); i++)
        {
            resNp.push_back((*this)[i] - npy[i]);
        }
        return resNp;
    }

    long double norm()
    {
        long double val = 0.0;
        for (auto element : (*this))
        {
            val += pow(element, 2);
        }
        return pow(val, 1.0 / 2);
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const numpy<T> &npy)
{
    try
    {
        os << "[";
        for (int i = 0; i < npy.size(); i++)
        {
            os << npy[i];
            if (i != npy.size() - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}
