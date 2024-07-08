using Accord.Collections;
using NeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace DrawingIdentifierGui.Models
{
    public class NeuralNetworkConfigModel
    {
        public bool SaveToLog { get; set; }
        public string SaveDirectoryPath { get; set; } = "";
        public bool SaveNeuralNetwork { get; set; }

        public bool IsPatience { get; set; }
        public float InitialIgnore { get; set; }
        public float Patience { get; set; }

        public (Matrix[] inputs, Matrix outputs)[] TrainData { get; set; } = new (Matrix[] inputs, Matrix outputs)[0];
        public (Matrix[] inputs, Matrix outputs)[] TestData { get; set; } = new (Matrix[] inputs, Matrix outputs)[0];

        public int SamplesPerFile { get; set; }
        public float InitialLearningRate { get; set; }
        public float MinLearningRate { get; set; }
        public int EpochAmount { get; set; }
        public int BatchSize { get; set; }

        public ObservableCollection<LayerModel> NeuralNetworkLayers { get; set; }
        public float? TestCorrectness { get; set; }
        public float? TrainCorrectness { get; set; }

        public static NeuralNetwork CreateNeuralNetwork(LayerModel[] layers)
        {
            if (layers[^1].LayerType != LayerType.FullyConnected || layers[^1].LayerSize != 10 || layers[^1].ActivationFunction != ActivationFunction.Softmax)
            {
                throw new ArgumentException("Last layer must be of type Fully Connected. It also need to have size of 10 and softmax as activation function");
            }

            int channels = 1;
            int rows = 28;
            int columns = 28;

            List<LayerTemplate> layerTemplates = new();

            foreach (var layer in layers)
            {
                switch (layer.LayerType)
                {
                    case LayerType.FullyConnected:
                        layerTemplates.Add(LayerTemplate.CreateFullyConnectedLayer(layer.LayerSize, layer.ActivationFunction));
                        break;

                    case LayerType.Convolution:
                        layerTemplates.Add(LayerTemplate.CreateConvolutionLayer(layer.KernelSize, layer.KernelDepth, 1, layer.ActivationFunction));
                        break;

                    case LayerType.Pooling:
                        layerTemplates.Add(LayerTemplate.CreateMaxPoolingLayer(layer.PoolSize, layer.PoolStride));
                        break;

                    case LayerType.Dropout:
                        layerTemplates.Add(LayerTemplate.CreateDropoutLayer((float)layer.DropoutRate));
                        break;
                }
            }

            return new NeuralNetwork(channels, rows, columns, layerTemplates.ToArray());
        }

        public NeuralNetwork CreateNeuralNetwork()
        {
            return NeuralNetworkConfigModel.CreateNeuralNetwork(NeuralNetworkLayers.ToArray());
        }

        public void LoadDataFromFile(string filePath)
        {
            ObservableCollection<LayerModel> res = new();

            XDocument xml = XDocument.Load(filePath);
            var root = xml.Root!;

            var head = root.Elements("LayersHead");
            foreach (var layerHead in head.First().Elements())
            {
                var layerTypeStr = layerHead.Attribute("LayerType")!.Value;
                LayerType layerType = Enum.Parse<LayerType>(layerTypeStr);

                switch (layerType)
                {
                    case LayerType.Convolution:
                        string? depthStr = layerHead.Element("depth")?.Value;
                        string? kernelSizeStr = layerHead.Element("kernelSize")?.Value;
                        string? strideStr = layerHead.Element("stride")?.Value;
                        string? activationFunctionStr = layerHead.Element("activationFunction")?.Value;


                        if (!int.TryParse(depthStr, out int depth) || !int.TryParse(kernelSizeStr, out int kernelSize) || !int.TryParse(strideStr, out int stride) || !Enum.TryParse<ActivationFunction>(activationFunctionStr, out ActivationFunction activationFunction))
                            throw new Exception();

                        res.Add(new LayerModel()
                        {
                            ActivationFunction = activationFunction,
                            KernelDepth = depth,
                            KernelSize = kernelSize,
                            LayerType = LayerType.Convolution,
                        });
                        break;
                    case LayerType.Pooling:
                        string? poolSizeStr = layerHead.Element("PoolSize")?.Value;
                        string? stridePoolStr = layerHead.Element("Stride")?.Value;
                        if (!int.TryParse(poolSizeStr, out int poolSize) || !int.TryParse(stridePoolStr, out int stridePool))
                            throw new Exception();

                        res.Add(new LayerModel()
                        {
                            LayerType = LayerType.Pooling,
                            PoolSize = poolSize,
                            PoolStride = stridePool,
                        });
                        break;
                    case LayerType.FullyConnected:
                        string? layerSizeStr = layerHead.Element("layerSize")?.Value;
                        string? activationFunctionFullStr = layerHead.Element("activationFunction")?.Value;


                        if (!int.TryParse(layerSizeStr, out int layerSize) || !Enum.TryParse<ActivationFunction>(activationFunctionFullStr, out ActivationFunction activationFunctionFull))
                            throw new Exception();

                        res.Add(new LayerModel()
                        {
                            LayerSize = layerSize,
                            ActivationFunction = activationFunctionFull,
                            LayerType = LayerType.FullyConnected,
                        });

                        break;
                    case LayerType.Dropout:
                        string? inputHeightStr = layerHead.Element("InputHeight")?.Value;
                        string? inputWidthStr = layerHead.Element("InputWidth")?.Value;
                        string? dropoutRateStr = layerHead.Element("DropoutRate")?.Value;

                        if (!int.TryParse(inputHeightStr, out int inputHeight) || !int.TryParse(inputWidthStr, out int inputWidth) || !float.TryParse(dropoutRateStr, out float dropoutRate))
                        {
                            throw new Exception();
                        }

                        res.Add(new LayerModel()
                        {
                            DropoutRate = dropoutRate,
                            LayerType = LayerType.Dropout,
                        });

                        break;
                    case LayerType.Reshape:
                        continue;
                    default:
                        throw new NotImplementedException();
                }
            }

            NeuralNetworkLayers = res;

            var config = root.Element("Config");
            if (config == null) throw new Exception("File damaged");

            var tmp = config.Element("TestCorrectness");
            if(tmp != null && float.TryParse(tmp.Value, out float correctness))
            {
                this.TestCorrectness = correctness; 
            }
            
            tmp = config.Element("LastTrainCorrectness");
            if(tmp != null && float.TryParse(tmp.Value, out correctness))
            { 
                this.TrainCorrectness = correctness;
            }
        }

        public Trainer CreateTrainer(NeuralNetwork neuralNetwork)
        {
            if (TrainData == null)
            {
                throw new InvalidOperationException("TrainData is null");
            }

            var trainer = new Trainer(neuralNetwork, TrainData, InitialLearningRate, MinLearningRate, EpochAmount, BatchSize);

            if (IsPatience)
            {
                trainer = trainer.SetPatience(Patience, InitialIgnore);
            }

            if (SaveToLog)
            {
                trainer = trainer.SetLogSaving(SaveDirectoryPath, SaveNeuralNetwork, testData: TestData, out _);
            }

            return trainer;
        }
    }
}