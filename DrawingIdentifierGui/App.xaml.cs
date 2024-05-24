using DrawingIdentifierGui.Models;
using NeuralNetworkLibrary;
using System.Configuration;
using System.Data;
using System.Windows;

namespace DrawingIdentifierGui
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    ///
    public partial class App : Application
    {
        public static ConvolutionalNeuralNetwork NeuralNetwork = new ConvolutionalNeuralNetwork([784, 16, 16, 10], [ActivationFunction.ReLU, ActivationFunction.ReLU, ActivationFunction.Softmax]);

        public static LearningConfig LearningConfigNN = new LearningConfig()
        {
            Data = null,
            LearningRate = 0.01,
            EpochAmount = 30,
            BatchSize = 50,
            ExpectedMaxError = 0.01
        };

        public static LearningConfig LearningConfigNNConvolutional = new LearningConfig()
        {
            Data = null,
            LearningRate = 0.01,
            EpochAmount = 30,
            BatchSize = 50,
            ExpectedMaxError = 0.01
        };
    }
}