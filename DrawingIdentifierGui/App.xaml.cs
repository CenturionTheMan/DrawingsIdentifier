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
        //TODO neural network aren t changing -> assign it somewhere (methods done in model)
        public static NeuralNetwork[] NeuralNetworks = [
            new NeuralNetwork(786,
                [
                    LayerTemplate.CreateFullyConnectedLayer(16, ActivationFunction.ReLU),
                    LayerTemplate.CreateFullyConnectedLayer(16, ActivationFunction.ReLU),
                    LayerTemplate.CreateFullyConnectedLayer(10, ActivationFunction.Softmax),
                ]
            ),

            new NeuralNetwork(1, 28, 28,
                [
                    LayerTemplate.CreateConvolutionLayer(5, 12, 1, ActivationFunction.ReLU),
                    LayerTemplate.CreateMaxPoolingLayer(2, 2),
                    LayerTemplate.CreateFullyConnectedLayer(64, ActivationFunction.ReLU),
                    LayerTemplate.CreateFullyConnectedLayer(64, ActivationFunction.ReLU),
                    LayerTemplate.CreateFullyConnectedLayer(10, ActivationFunction.Softmax),
                ]
            ),
        ];

        public static NeuralNetworkConfigModel[] NeuralNetworkConfigModels = [
            new NeuralNetworkConfigModel()
            {
                TrainData = null,
                TestData = null,
                SamplesPerFile = 5000,
                InitialLearningRate = 0.01f,
                MinLearningRate = 0.001f,
                EpochAmount = 30,
                BatchSize = 50,
                NeuralNetworkLayers = new()
                {
                    new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                    new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                }
            },
            new NeuralNetworkConfigModel()
            {
                TrainData = null,
                TestData = null,
                SamplesPerFile = 5000,
                InitialLearningRate = 0.01f,
                MinLearningRate = 0.001f,
                EpochAmount = 30,
                BatchSize = 50,
                NeuralNetworkLayers = new()
                {
                    new LayerModel() { LayerType = LayerType.Convolution, KernelSize = 5, KernelDepth = 8},
                    new LayerModel() { LayerType = LayerType.MaxPooling, PoolSize = 2, PoolStride = 2},
                    new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                    new LayerModel() { LayerType = LayerType.Dropout, DropoutRate = 0.5f},
                    new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                }
            },
        ];
    }
}