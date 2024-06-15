using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using static NeuralNetworkLibrary.ActivationFunctionsHandler;

namespace NeuralNetworkLibrary;

public class ConvolutionLayer : ILayer
{
    LayerType ILayer.LayerType => LayerType.Convolution;

    // public int Depth => depth;
    // public int KernelSize => kernelSize;
    // public int Stride => stride;

    private int depth;
    private int kernelSize;
    private int stride;

    private ActivationFunction activationFunction;

    private Matrix[,] kernels;
    private Matrix[] biases;

    private Matrix[,] changeForKernels;
    private Matrix[] changeForBiases;

    private int inputDepth;
    private int inputWidth;
    private int inputHeight;

    public ConvolutionLayer((int inputDepth, int inputHeight, int inputWidth) inputShape, int kernelSize, int kernelsDepth, int stride, ActivationFunction activationFunction, double minInitValue = -0.2, double maxInitValue = 0.2)
    {
        if(stride != 1)
        {
            throw new NotImplementedException("Stride != 1 is not implemented yet");
        }

        if(stride < 1)
        {
            throw new ArgumentException("Stride must be greater than 0");
        }

        this.depth = kernelsDepth;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.activationFunction = activationFunction;

        this.inputDepth = inputShape.inputDepth;
        this.inputWidth = inputShape.inputWidth;
        this.inputHeight = inputShape.inputHeight;

        kernels = new Matrix[depth, inputShape.inputDepth];
        biases = new Matrix[depth];

        changeForKernels = new Matrix[depth, inputShape.inputDepth];
        changeForBiases = new Matrix[depth];


        (int outputRows, int outputColumns) = MatrixExtender.GetSizeAfterConvolution((inputHeight, inputWidth), (kernelSize, kernelSize), stride);
        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < inputShape.inputDepth; j++)
            {
                kernels[i, j] = new Matrix(kernelSize, kernelSize, minInitValue, maxInitValue);
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }

            biases[i] = new Matrix(outputRows, outputColumns, minInitValue, maxInitValue);
            changeForBiases[i] = new Matrix(outputRows, outputColumns);
        }
    }

    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        // activated output
        Matrix[] A = new Matrix[depth];
        // output before activation
        Matrix[] Z = new Matrix[depth];
        
        for (int i = 0; i < depth; i++)
        {
            A[i] = new Matrix(biases[i].RowsAmount, biases[i].ColumnsAmount);
            Z[i] = new Matrix(biases[i].RowsAmount, biases[i].ColumnsAmount);
        }

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                var single = inputs[j].CrossCorrelationValid(kernels[i, j], stride: this.stride);
                Z[i] = Z[i].ElementWiseAdd(single);
            }
            Z[i] = Z[i].ElementWiseAdd(biases[i]);

            A[i] = Z[i].ApplyActivationFunction(activationFunction);
        }

        return (A, Z);
    }

    Matrix[] ILayer.Backward(Matrix[] dAin, Matrix[] layerInputFromForward, Matrix[] layerOutputBeforeActivation, double learningRate)
    {
        Matrix[] dA = new Matrix[inputDepth];
        for (int i = 0; i < inputDepth; i++)
        {
            dA[i] = new Matrix(inputHeight, inputWidth);
        }

        Matrix[] dZ = new Matrix[layerOutputBeforeActivation.Length];
        for (int i = 0; i < layerOutputBeforeActivation.Length; i++)
        {
            dZ[i] = Matrix.ElementWiseMultiplyMatrices(dAin[i], layerOutputBeforeActivation[i].DerivativeActivationFunction(activationFunction)); 
        }

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < inputDepth; j++)
            {
                Matrix kernelGradient = layerInputFromForward[j].CrossCorrelationValid(dZ[i], stride: this.stride);
                kernelGradient = kernelGradient * learningRate;
                changeForKernels[i, j] = changeForKernels[i, j].ElementWiseAdd(kernelGradient);

                var dASingle = dZ[i].ConvolutionFull(kernels[i, j], stride: this.stride);
                dA[j] = dA[j].ElementWiseAdd(dASingle);
            }

            changeForBiases[i] = changeForBiases[i].ElementWiseAdd(dZ[i] * learningRate);
        }

        return dA;
    }

    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        double multiplier = 1.0 / (double)batchSize;

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < kernels.GetLength(1); j++)
            {
                kernels[i, j] = kernels[i, j].ElementWiseAdd(changeForKernels[i, j] * multiplier);
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }
            biases[i] = biases[i].ElementWiseAdd(changeForBiases[i] * multiplier);
            changeForBiases[i] = new Matrix(biases[i].RowsAmount, biases[i].ColumnsAmount);
        }
    }
}