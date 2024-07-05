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
        public static NeuralNetwork[] NeuralNetworks;
        public static NeuralNetworkConfigModel[] NeuralNetworkConfigModels;

        static App()
        {
            NeuralNetworkConfigModels = [
                new NeuralNetworkConfigModel()
                {
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

            NeuralNetworks = new NeuralNetwork[NeuralNetworkConfigModels.Length];
            for (int i = 0; i < NeuralNetworkConfigModels.Length; i++)
            {
                NeuralNetworks[i] = NeuralNetworkConfigModels[i].CreateNeuralNetwork();
            }
        }
    }


}