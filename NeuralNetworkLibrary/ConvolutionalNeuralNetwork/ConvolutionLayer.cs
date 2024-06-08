using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NeuralNetworkLibrary.Utilities;

namespace NeuralNetworkLibrary;

public class ConvolutionLayer : IFeatureExtractionLayer
{
    private static readonly Random random = new Random();

    private Matrix[] kernels;
    private double[] biases;
    private int stride;
    private ActivationFunction activationFunction;
    private int padding;

    private Matrix[] changeForKernels;
    // private double[] changeForBiases;

    public ConvolutionLayer(int kernelsAmount, int kernelRows, int kernelColumns, ActivationFunction activationFunction, int stride = 1, int padding = 1, double minInitValue = -0.1, double maxInitValue = 0.1)
    {
        this.padding = padding;
        this.activationFunction = activationFunction;
        kernels = new Matrix[kernelsAmount];
        biases = new double[kernelsAmount];
        changeForKernels = new Matrix[kernelsAmount];
        // changeForBiases = new double[kernelsAmount];

        for (int i = 0; i < kernelsAmount; i++)
        {
            kernels[i] = new Matrix(kernelRows, kernelColumns, minInitValue, maxInitValue);
            biases[i] = random.NextDouble() * (maxInitValue - minInitValue) + minInitValue;
            changeForKernels[i] = new Matrix(kernelRows, kernelColumns);
        }

        this.stride = stride;
    }

    Matrix[] IFeatureExtractionLayer.Forward(Matrix[] inputs)
    {
        List<Matrix> results = new List<Matrix>(kernels.Length);
        
        List<Matrix> paddedInputs = new List<Matrix>(inputs.Length);
        foreach (var input in inputs)
        {
            paddedInputs.Add(PadInput(input, padding));
        }
        
        foreach (var kernel in kernels)
        {
            Matrix sumOfConvolution = new Matrix(inputs[0].RowsAmount, inputs[0].ColumnsAmount);
            foreach (var paddedInput in paddedInputs)
            {
                Matrix result = Convolve(paddedInput, kernel, stride);
                sumOfConvolution = sumOfConvolution.ElementWiseAdd(result);
            }

            results.Add(ApplyActivationFunction(sumOfConvolution));
        }
        return results.ToArray();
    }

    Matrix[] IFeatureExtractionLayer.Backward(Matrix[] errors, Matrix[] previousLayerOutputs, double learningRate)
    {
        List<Matrix> changesFor = new List<Matrix>(kernels.Length);
        List<Matrix> gradients = new List<Matrix>(kernels.Length);
            

        List<Matrix> paddedInputs = new List<Matrix>(previousLayerOutputs.Length);
        foreach (var prev in previousLayerOutputs)
        {
            paddedInputs.Add(PadInput(prev, padding));
        }

        for (int i = 0; i < kernels.Length; i++)
        {
            Matrix dout = new Matrix(errors[0].RowsAmount, errors[0].ColumnsAmount);
            for (int j = 0; j < errors.Length; j++)
            {
                var err = DerivativeActivationFunction(errors[i]);
                var tmp = Convolve(paddedInputs[i], Rotate180(err), stride);

                changeForKernels[i] = changeForKernels[i].ElementWiseSubtract(Matrix.ElementWiseMultiplyMatrices(err, tmp) * learningRate);

                dout = dout.ElementWiseAdd(Matrix.ElementWiseMultiplyMatrices(err, kernels[i]));
            }

            gradients.Add(dout);
        }

        return gradients.ToArray();
    }

    public void UpdateWeightsAndBiases(double batchSize)
    {
        for (int i = 0; i < kernels.Length; i++)
        {
            kernels[i] = kernels[i].ElementWiseAdd(changeForKernels[i].ApplyFunction(x => x / batchSize));
            //biases[i] += changeForBiases[i] / batchSize;

            changeForKernels[i] = new Matrix(kernels[i].RowsAmount, kernels[i].ColumnsAmount);
            //changeForBiases[i] = 0;
        }
    }


    private Matrix PadInput(Matrix input, int padding)
    {
        Matrix paddedInput = new Matrix(input.RowsAmount + 2 * padding, input.ColumnsAmount + 2 * padding);
        for (int i = 0; i < input.RowsAmount; i++)
        {
            for (int j = 0; j < input.ColumnsAmount; j++)
            {
                paddedInput[i + padding, j + padding] = input[i, j];
            }
        }
        return paddedInput;
    }

    private Matrix Convolve(Matrix input, Matrix kernel, int stride)
    {
        int outputRows = ((input.RowsAmount - kernel.RowsAmount) / stride) + 1;
        int outputColumns = ((input.ColumnsAmount - kernel.ColumnsAmount) / stride) + 1;
        Matrix output = new Matrix(outputRows, outputColumns);

        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputColumns; j++)
            {
                double sum = 0;

                for (int m = 0; m < kernel.RowsAmount; m++)
                {
                    for (int n = 0; n < kernel.ColumnsAmount; n++)
                    {
                        int x = i * stride + m;
                        int y = j * stride + n;

                        if (x >= 0 && x < input.RowsAmount && y >= 0 && y < input.ColumnsAmount)
                        {
                            sum += input[x, y] * kernel[m, n];
                        }
                    }
                }

                output[i, j] = sum;
            }
        }
        return output;
    }

    //TODO TEST
    /// <summary>
    /// Rotates the matrix by 180 degrees
    /// </summary>
    /// <returns>New matrix after rotation</returns>
     private Matrix Rotate180(Matrix matrix)
    {
        Matrix result = new Matrix(matrix.RowsAmount, matrix.ColumnsAmount);
        for (int i = 0; i < matrix.RowsAmount; i++)
        {
            for (int j = 0; j < matrix.ColumnsAmount; j++)
            {
                result[i, j] = matrix[matrix.RowsAmount - i - 1, matrix.ColumnsAmount - j - 1];
            }
        }
        return result;
    }

    private Matrix ApplyActivationFunction(Matrix input)
    {
        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                return Utilities.ReLU(input);
            case ActivationFunction.Sigmoid:
                return Utilities.Sigmoid(input);
            case ActivationFunction.Softmax:
                return Utilities.Softmax(input);
            default:
                throw new ArgumentException("Invalid activation function");
        }
    }

    private Matrix DerivativeActivationFunction(Matrix input)
    {
        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                return Utilities.DerivativeReLU(input);
            case ActivationFunction.Sigmoid:
                return Utilities.DerivativeSigmoid(input);
            case ActivationFunction.Softmax:
                return Utilities.DerivativeSoftmax(input);
            default:
                throw new ArgumentException("Invalid activation function");
        }
    }

}