using DrawingIdentifierGui.Models;
using NeuralNetworkLibrary;
using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.NeuralNetwork;
using NeuralNetworkLibrary.Utils;
using System.Windows;

namespace DrawingIdentifierGui
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    ///
    public partial class App : Application
    {
        public const int CLASSES_AMOUNT = 9;

        private static string[] initNNPaths = new string[]
        {
            "NN1.xml",
            "NN2.xml",
        };

        public static NeuralNetwork[] NeuralNetworks = new NeuralNetwork[2];
        public static NeuralNetworkConfigModel[] NeuralNetworkConfigModels = new NeuralNetworkConfigModel[2];

        public static (NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)[] TrainData { get; set; } = Array.Empty<(NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)>();
        public static (NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)[] TestData { get; set; } = Array.Empty<(NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)>();

        public static (NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)[] TrainDataFlat { get; set; } = Array.Empty<(NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)>();
        public static (NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)[] TestDataFlat { get; set; } = Array.Empty<(NeuralNetworkLibrary.Math.Matrix[] inputs, NeuralNetworkLibrary.Math.Matrix outputs)>();

        public static bool IsExampleNN1Loaded;
        public static bool IsExampleNN2Loaded;

        static App()
        {
            App.IsExampleNN1Loaded = TryLoadInitialNN(0);
            App.IsExampleNN2Loaded = TryLoadInitialNN(1);

            if(!IsExampleNN1Loaded)
            {
                NeuralNetworkConfigModels[0] = new NeuralNetworkConfigModel()
                {
                    InitialLearningRate = 0.01f,
                    MinLearningRate = 0.001f,
                    EpochAmount = 30,
                    BatchSize = 50,
                    NeuralNetworkLayers = new()
                    {
                        new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                        new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                        new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = App.CLASSES_AMOUNT, ActivationFunction = ActivationFunction.Softmax},
                    }
                };

                NeuralNetworks[0] = NeuralNetworkConfigModels[0].CreateNeuralNetwork();
            }

            if(!IsExampleNN2Loaded)
            {
                NeuralNetworkConfigModels[1] = new NeuralNetworkConfigModel()
                {
                    InitialLearningRate = 0.01f,
                    MinLearningRate = 0.001f,
                    EpochAmount = 30,
                    BatchSize = 50,
                    NeuralNetworkLayers = new()
                    {
                        new LayerModel() { LayerType = LayerType.Convolution, KernelSize = 5, KernelDepth = 8},
                        new LayerModel() { LayerType = LayerType.Pooling, PoolSize = 2, PoolStride = 2},
                        new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                        new LayerModel() { LayerType = LayerType.Dropout, DropoutRate = 0.5f},
                        new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = 16, ActivationFunction = ActivationFunction.ReLU},
                        new LayerModel() { LayerType = LayerType.FullyConnected, LayerSize = App.CLASSES_AMOUNT, ActivationFunction = ActivationFunction.Softmax},
                    }
                };

                NeuralNetworks[1] = NeuralNetworkConfigModels[1].CreateNeuralNetwork();
            }

            
        }

        private static bool TryLoadInitialNN(int index)
        {
            var nn = NeuralNetwork.LoadFromXmlFile(initNNPaths[index]);
            if (nn is null) return false;

            try
            {
                NeuralNetworkConfigModels[index] = new NeuralNetworkConfigModel();
                NeuralNetworkConfigModels[index].LoadDataFromFile(initNNPaths[index]);
                NeuralNetworks[index] = nn;
                return true;
            }
            catch
            {
                return false;
            }
        }

    }
}