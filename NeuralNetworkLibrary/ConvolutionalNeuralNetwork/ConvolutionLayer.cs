using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using static NeuralNetworkLibrary.ActivationFunctionsHandler;

namespace NeuralNetworkLibrary;

public class ConvolutionLayer : IFeatureExtractionLayer
{
    public int Depth => depth;
    public int KernelSize => kernelSize;
    public ActivationFunction ActivationFunction => activationFunction;
    public int Stride => 1;

    private int depth;
    private int kernelSize;

    private ActivationFunction activationFunction;

    private Matrix[,] kernels;
    private Matrix[] biases;
    //private ActivationFunction activationFunction;

    private Matrix[,] changeForKernels;
    private Matrix[] changeForBiases;

    private int inputDepth;
    private int inputWidth;
    private int inputHeight;

    private double minInitValue;
    private double maxInitValue;

    public ConvolutionLayer(int kernelSize, int kernelsDepth, ActivationFunction activationFunction, double minInitValue = -0.1, double maxInitValue = 0.1)
    {
        this.depth = kernelsDepth;
        this.kernelSize = kernelSize;
        this.activationFunction = activationFunction;
        this.minInitValue = minInitValue;
        this.maxInitValue = maxInitValue;

        this.inputDepth = -1;
        this.inputWidth = -1;
        this.inputHeight = -1;

        this.kernels = new Matrix[0, 0];
        this.biases = new Matrix[0];

        this.changeForKernels = new Matrix[0, 0];
        this.changeForBiases = new Matrix[0];
    }

    void IFeatureExtractionLayer.Initialize((int inputDepth, int inputHeight, int inputWidth) inputShape)
    {
        this.inputDepth = inputShape.inputDepth;
        this.inputWidth = inputShape.inputWidth;
        this.inputHeight = inputShape.inputHeight;

        kernels = new Matrix[depth, inputShape.inputDepth];
        biases = new Matrix[depth];

        changeForKernels = new Matrix[depth, inputShape.inputDepth];
        changeForBiases = new Matrix[depth];

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < inputShape.inputDepth; j++)
            {
                kernels[i, j] = new Matrix(kernelSize, kernelSize, minInitValue, maxInitValue);
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }
            biases[i] = new Matrix(inputShape.inputHeight - kernelSize + 1, inputShape.inputWidth - kernelSize + 1, minInitValue, maxInitValue);
            changeForBiases[i] = new Matrix(inputShape.inputHeight - kernelSize + 1, inputShape.inputWidth - kernelSize + 1);
        }
    }

    (Matrix[] output, Matrix[] outputsBeforeActivation) IFeatureExtractionLayer.Forward(Matrix[] inputs)
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
                var single = inputs[j].CrossCorrelationValid(kernels[i, j], stride: 1);
                Z[i] = Z[i].ElementWiseAdd(single);
            }
            Z[i] = Z[i].ElementWiseAdd(biases[i]);

            A[i] = Z[i].ApplyActivationFunction(activationFunction);
        }

        return (A, Z);
    }

    Matrix[] IFeatureExtractionLayer.Backward(Matrix[] dAin, Matrix[] layerInputFromForward, Matrix[] layerOutputBeforeActivation, double learningRate)
    {
        //output gradient
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
                Matrix kernelGradient = layerInputFromForward[j].CrossCorrelationValid(dZ[i], stride: 1);
                kernelGradient = kernelGradient * learningRate;
                changeForKernels[i, j] = changeForKernels[i, j].ElementWiseAdd(kernelGradient);

                // var inputGradientSingle = inputGradient[i].ConvolutionFull(kernels[i, j], stride: 1);
                // outputGradient[j] = outputGradient[j].ElementWiseAdd(inputGradientSingle);

                var dASingle = dZ[i].ConvolutionFull(kernels[i, j], stride: 1);
                dA[j] = dA[j].ElementWiseAdd(dASingle);
            }

            changeForBiases[i] = changeForBiases[i].ElementWiseAdd(dZ[i] * learningRate);
        }

        return dA;
    }

    public void UpdateWeightsAndBiases(double batchSize)
    {
        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < kernels.GetLength(1); j++)
            {
                kernels[i, j] = kernels[i, j].ElementWiseAdd(changeForKernels[i, j] * (1.0 / batchSize));
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }
            biases[i] = biases[i].ElementWiseAdd(changeForBiases[i] * (1.0 / batchSize));
            changeForBiases[i] = new Matrix(biases[i].RowsAmount, biases[i].ColumnsAmount);
        }
    }
}