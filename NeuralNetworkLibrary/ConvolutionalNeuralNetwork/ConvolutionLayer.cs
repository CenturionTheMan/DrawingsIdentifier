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

    public ConvolutionLayer((int inputDepth, int inputWidth, int inputHeight) inputShape, int kernelSize, int kernelsDepth, ActivationFunction activationFunction, double minInitValue = -0.1, double maxInitValue = 0.1)
    {
        this.inputDepth = inputShape.inputDepth;
        this.inputWidth = inputShape.inputWidth;
        this.inputHeight = inputShape.inputHeight;

        this.depth = kernelsDepth;
        this.kernelSize = kernelSize;

        this.activationFunction = activationFunction;

        kernels = new Matrix[kernelsDepth, inputShape.inputDepth];
        biases = new Matrix[kernelsDepth];

        changeForKernels = new Matrix[kernelsDepth, inputShape.inputDepth];
        changeForBiases = new Matrix[kernelsDepth];

        for (int i = 0; i < kernelsDepth; i++)
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
        Matrix[] outputs = new Matrix[depth];
        Matrix[] outputsBeforeActivation = new Matrix[depth];

        for (int i = 0; i < depth; i++)
        {
            outputs[i] = Matrix.Copy(biases[i]);
            
            for (int j = 0; j < inputs.Length; j++)
            {
                var single = inputs[j].CrossCorrelationValid(kernels[i, j], stride: 1);
                outputs[i] = outputs[i].ElementWiseAdd(single);
            }

            outputsBeforeActivation[i] = Matrix.Copy(outputs[i]);
            outputs[i] = outputs[i].ApplyActivationFunction(activationFunction);
        }

        return (outputs, outputsBeforeActivation);
    }

    Matrix[] IFeatureExtractionLayer.Backward(Matrix[] inputGradient, Matrix[] previousLayerOutputs, double learningRate)
    {
        Matrix[,] kernelsGradient = new Matrix[this.depth, previousLayerOutputs.Length];
        for (int i = 0; i < kernelsGradient.GetLength(0); i++)
        {
            for (int j = 0; j < kernelsGradient.GetLength(1); j++)
            {
                kernelsGradient[i, j] = new Matrix(kernelSize, kernelSize);
            }
        }

        Matrix[] outputGradient = new Matrix[inputDepth];
        for (int i = 0; i < previousLayerOutputs.Length; i++)
        {
            outputGradient[i] = new Matrix(inputHeight, inputWidth);
        }

        inputGradient = inputGradient.Select(x => x = x.DerivativeActivationFunction(activationFunction)).ToArray();

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < previousLayerOutputs.Length; j++)
            {
                Matrix kernelGradient = previousLayerOutputs[j].CrossCorrelationValid(inputGradient[i], stride: 1);
                changeForKernels[i, j] = changeForKernels[i, j].ElementWiseSubtract(kernelGradient * learningRate);

                var inputGradientSingle =inputGradient[i].ConvolutionFull(kernels[i, j], stride: 1);
                outputGradient[j] = outputGradient[j].ElementWiseAdd(inputGradientSingle);
            }

            changeForBiases[i] = changeForBiases[i].ElementWiseSubtract(inputGradient[i]  * learningRate);
        }

        return outputGradient;
    }

    public void UpdateWeightsAndBiases(double batchSize)
    {
        for(int i = 0; i < depth; i++)
        {
            for(int j = 0; j < kernels.GetLength(1); j++)
            {
                kernels[i, j] = kernels[i, j].ElementWiseAdd(changeForKernels[i, j] * (1.0 / batchSize));
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }
            biases[i] = biases[i].ElementWiseAdd(changeForBiases[i] * (1.0 / batchSize));
            changeForBiases[i] = new Matrix(biases[i].RowsAmount, biases[i].ColumnsAmount);
        }
    }
   

}