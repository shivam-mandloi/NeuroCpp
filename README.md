# Recurrent Neural Network (RNN) in C++ and Python

## Overview
This project implements a Recurrent Neural Network (RNN) using pure C++ without external libraries. However, Python (NumPy) is used for additional processing. The implementation includes modular components for activation functions, loss functions, and optimizers, making it easy to define various neural network architectures.

## Project Structure
Each block of the neural network is implemented as a separate class:
- **Optimizers**: `Adam`, `SGD`
- **Activation Functions**: `ReLU`, `Softmax`
- **Loss Function**: `CrossEntropyLoss`
- **Neurons and Layers**: Defined as `Neuron` classes

Each block has its own `Forward` and `BackPropagate` methods, allowing easy composition of feedforward neural networks (FNNs).

## Example: Feedforward Neural Network (FNN) for Iris Dataset

```cpp
struct nn {
    Neuron nn1;
    Relu rl1;
    Neuron nn2;
    Relu rl2;
    Neuron nn3;
    Softmax sf;
    CrossEntropy ls;

    nn() : nn1(4, 10, SGDOptim, 0.0001), nn2(10, 10), nn3(10, 3, SGDOptim, 0.1) {}

    numpy<double> Forward(numpy<double> input) {
        return sf.Forward(nn3.Forward(rl2.Forward(nn2.Forward(rl1.Forward(nn1.Forward(input))))));
    }

    void BackPropagate(numpy<double> pred, numpy<double> actual) {
        nn1.BackPropagate(rl1.BackPropagate(nn2.BackPropagate(rl2.BackPropagate(nn3.BackPropagate(sf.BackPropagate(ls.BackPropagate(pred, actual)))))));
    }

    double loss(numpy<double> input, int index, bool check = 1) {
        numpy<double> actual(input.size(), 0);
        actual[index] = 1;
        std::cout << "predicted: " << input << " Actual: " << actual << std::endl;
        if (check)
            BackPropagate(input, actual);
        return ls.Forward(input, actual);
    }
};
```

## Compilation & Execution

1. **Organizing Files**
   - Place all your C++ files in the `main.cpp` directory.
   - You may create header files but ensure they are in the same folder as `main.cpp`.
   - If using subfolders, ensure they are inside the directory containing `main.cpp`.

2. **Running the Model**
   - Use the `script.py` file to compile and run your code.
   - The script will iterate through all directories and include relevant files.

3. **Executing the Script**
   Run the following command:
   ```bash
   python script.py
   ```

   If successful, you will see:
   ```
   [*] Compilation successful!
   File Name: full\path\main.cpp
   ```
   
   This generates an executable `main.exe` file.

4. **Customization**
   - Modify `script.py` to fit your project structure and compilation needs.

## Notes
- Ensure C++ and Python (with NumPy) are installed on your system.
- Adapt `script.py` based on additional dependencies or different directory structures.
- Modify neural network parameters (`nn1`, `nn2`, etc.) as needed for your dataset.

## Future Enhancements
- Adding support for more activation functions and optimizers.
- Implementing RNN, LSTM, and Transformer models using the same modular approach.
- Parallelizing computations using multi-threading for faster training.

## License
This project is open-source. Feel free to modify and use it as needed!
