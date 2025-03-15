#include "numpy.hpp"

class NpHelpingFunc
{
public:
    numpy<double> MatMul(numpy<numpy<double>> mat, numpy<double> vec)
    {
        if(mat.size() > 0 && mat[0].size() != vec.size())
        {
            std::cerr << "size not match matrix multiplication " << "NumpyHelpingFunc.hpp/MatMul" << std::endl;
            return numpy<double>();
        }
        numpy<double> res;
        for(int i = 0; i < mat.size(); i++)
        {
            double temp = 0;
            for(int j = 0; j < mat[i].size(); j++)
            {
                temp += (mat[i][j] * vec[j]);
            }
            res.push_back(temp);
        }
        return res;
    }
};