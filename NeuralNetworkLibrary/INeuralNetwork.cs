using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public interface INeuralNetwork
{
    public void Train((double[] inputs, double[] outputs)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001);

    public Action<int, double, double>? OnLearningIteration { get; set; }

    public double[] Predict(double[] inputs);
}