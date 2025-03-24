#pragma once

#include <cstdlib>
#include <string>
#include "numpy.hpp"
#include <fstream>

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

    numpy<double> SplitString(std::string str)
    {
        /*
            take string a = "1 2 3 4 5" and return numpy<double> arr = {1, 2, 3, 4, 5} type = double
        */
        numpy<double> arr;
        std::string temp;
        for(int i = 0; i < str.size(); i++)
        {
            if(str[i] == ' ')
            {
                if(temp != " " && temp != "")
                    arr.push_back(std::stod(temp));
                temp = "";
                continue;
            }
            temp += str[i];
        }
        if(temp != "" && temp != " ")
            arr.push_back(std::stod(temp));
        return arr;
    }

    numpy<numpy<double>> rand(int a, int b, std::string fileName = "weight.txt")
    {
        //  matrix shape => a X b

        // initialize the mat numpy vector
        numpy<numpy<double>> mat(a, numpy<double>(b, 0.0));

        // Run python command
        std::string cmd = "python Include\\Numpy\\ParameterInitializer.py " + std::to_string(a) + " " + std::to_string(b) + " " + fileName;
        if(system(cmd.c_str())) // if system(cmd.c_str() != 0) then give error
        {
            std::cout << "[!] Parameter Initialization problem" << std::endl; 
            exit(0);
        }

        // open file
        std::fstream newFile;
        newFile.open(fileName, std::ios::in);
        std::string temp;

        if (!newFile.is_open())
        {
            std::cerr << "Error: Could not open file " << fileName << std::endl;
            exit(0);
        }
        
        int index = 0;
        // Read Data in temp variable
        while (getline(newFile, temp))
        {
            if (temp != "")
            {
                mat[index] = SplitString(temp);
                index += 1;
            }
        }
        return mat;
    }

    numpy<double> rand(int a, std::string fileName = "bias.txt")
    {
        std::fstream newFile;
        std::string temp;
        numpy<double> vec(a, 0);

        std::string cmd = "python Include\\Numpy\\ParameterInitializer.py " + std::to_string(a) + " " + fileName;
        if(system(cmd.c_str())) // if system(cmd.c_str() != 0) then give error
        {
            std::cout << "[!] Parameter Initialization problem" << std::endl; 
            exit(0);
        }

        newFile.open(fileName, std::ios::in); // open bias.txt file
        if (!newFile.is_open())
        {
            std::cerr << "Error: Could not open file " << fileName << std::endl;
            exit(0);
        }
        int index = 0;
        while (getline(newFile, temp))
        {
            if (temp != "")
            {
                vec[index] = std::stod(temp);
                index += 1;
            }
        }

        return vec;
    }

    std::string ReadFile(std::string path)
    {
        std::fstream newFile;
        std::string temp;
        std::string data = "";

        newFile.open(path, std::ios::in);
        if (!newFile.is_open())
        {
            std::cerr << "Error: Could not open file " << path << std::endl;
            exit(0);
        }

        while (getline(newFile, temp))
        {
            if (temp != "")
                data += temp + " ";
        }

        return data;
    }
};