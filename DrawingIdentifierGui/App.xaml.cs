using DrawingIdentifierGui.Models;
using NeuralNetworkLibrary;
using System.Collections.ObjectModel;
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
        public static FeedForwardNeuralNetwork FeedForwardNN = new FeedForwardNeuralNetwork([784, 16, 16, 10], [ActivationFunction.ReLU, ActivationFunction.ReLU, ActivationFunction.Softmax]);

        public static LearningConfig FeedForwardNNConfig = new LearningConfig()
        {
            Data = null,
            SamplesAmountToLoadPerFile = 5000,
            LearningRate = 0.01,
            EpochAmount = 30,
            BatchSize = 50,
            ExpectedMaxError = 0.01,
            NeuralNetworkLayers = new()
            {
                new NNLayerConfig { LayerName="Input Layer", Size=784, ActivationFunction=ActivationFunction.ReLU, IsSizeEnable=false },
                new NNLayerConfig { LayerName="Hidden Layer", Size=16, ActivationFunction=ActivationFunction.ReLU },
                new NNLayerConfig { LayerName="Hidden Layer", Size=16, ActivationFunction=ActivationFunction.Softmax, IsActivationFunctionEnable=false },
                new NNLayerConfig { LayerName="Output Layer", Size=10, ActivationFunction=null, IsSizeEnable=false, IsActivationFunctionEnable=false },
            }
        };

        public static LearningConfig ConvolutionalNNConfig = new LearningConfig()
        {
            Data = null,
            LearningRate = 0.01,
            EpochAmount = 30,
            BatchSize = 50,
            ExpectedMaxError = 0.01
        };
    }
}