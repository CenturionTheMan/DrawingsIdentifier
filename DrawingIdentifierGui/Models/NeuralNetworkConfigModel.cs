using Accord.Collections;
using NeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

        public static NeuralNetwork CreateNeuralNetwork(LayerModel[] layers)
        {
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

                    case LayerType.MaxPooling:
                        layerTemplates.Add(LayerTemplate.CreateMaxPoolingLayer(layer.PoolSize, layer.PoolStride));
                        break;

                    case LayerType.Dropout:
                        layerTemplates.Add(LayerTemplate.CreateDropoutLayer((float)layer.DropoutRate));
                        break;
                }
            }
            layerTemplates.Add(LayerTemplate.CreateFullyConnectedLayer(10, ActivationFunction.Softmax));

            return new NeuralNetwork(channels, rows, columns, layerTemplates.ToArray());
        }

        public NeuralNetwork CreateNeuralNetwork()
        {
            return NeuralNetworkConfigModel.CreateNeuralNetwork(NeuralNetworkLayers.ToArray());
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
                trainer = trainer.SetLogSaving(SaveDirectoryPath, SaveNeuralNetwork, out _);
            }

            return trainer;
        }
    }
}