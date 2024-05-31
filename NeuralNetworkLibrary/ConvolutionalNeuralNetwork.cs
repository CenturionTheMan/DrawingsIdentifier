using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public class ConvolutionalNeuralNetwork : INeuralNetwork
{
    public Action<int, double, double>? OnLearningIteration { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

    public bool IsStructureEqual(int[] layersSize, ActivationFunction[] activationFunctions)
    {
        throw new NotImplementedException();
    }

    public double[] Predict(double[] inputs)
    {
        throw new NotImplementedException();
    }

    public void Train((double[] inputs, double[] outputs)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001)
    {
        throw new NotImplementedException();
    }
}