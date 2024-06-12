using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using MyBaseLibrary;

namespace NeuralNetworkLibrary;

public class ConvolutionalNeuralNetwork
{
    private const bool multiThreaded = true;

    public bool saveFeatureLayersOutputs = false;

    private const string logFilePath = "D:\\GoogleDriveMirror\\Studia\\Inzynierka\\LearningLogs\\log";
    private const bool logToFile = false;

    private static Random random = new Random();

    private IFeatureExtractionLayer[] featureLayers;
    private FullyConnectedLayer[] fullyConnectedLayers;
    private double learningRate;

    private (int rows, int columns) outputFromLastFeatureLayerSize;

    public ConvolutionalNeuralNetwork((int depth, int rows, int columns) input, IFeatureExtractionLayer[] featureExtractionLayers, FullyConnectedLayer[] fullyConnectedLayers)
    {
        this.featureLayers = featureExtractionLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;

        if (featureExtractionLayers.Length > 0)
        {
            featureExtractionLayers[0].Initialize(input);
            var size = Utilities.GetSizeAfterConvolution((input.rows, input.columns), (featureExtractionLayers[0].KernelSize, featureExtractionLayers[0].KernelSize), featureExtractionLayers[0].Stride);

            for (int i = 1; i < featureExtractionLayers.Length; i++)
            {
                featureExtractionLayers[i].Initialize((featureExtractionLayers[i - 1].Depth, size.outputRows, size.outputColumns));
                size = Utilities.GetSizeAfterConvolution((size.outputRows, size.outputColumns), (featureExtractionLayers[i].KernelSize, featureExtractionLayers[i].KernelSize), featureExtractionLayers[i].Stride);
            }
            outputFromLastFeatureLayerSize = (size.outputRows, size.outputColumns);
        }

        if (fullyConnectedLayers.Length > 0)
        {
            int depth = featureExtractionLayers.Length > 0 ? featureExtractionLayers.Last().Depth : input.depth;
            fullyConnectedLayers[0].Initialize(outputFromLastFeatureLayerSize.rows * outputFromLastFeatureLayerSize.columns * depth);
            for (int i = 1; i < fullyConnectedLayers.Length; i++)
            {
                fullyConnectedLayers[i].Initialize(fullyConnectedLayers[i - 1].LayerSize);
            }
        }
    }

    public void Train((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001)
    {
        var guid = Guid.NewGuid();

        if (logToFile)
        {
            var writer = FilesCreatorHelper.CreateXmlFile(logFilePath + "_" + guid.ToString() + ".xml");
            writer.WriteStartElement("Root");
            writer.WriteStartElement("LearningInstance");
            writer.WriteElementString("Guid", guid.ToString());
            writer.WriteElementString("LearningRate", learningRate.ToString());
            writer.WriteElementString("EpochAmount", epochAmount.ToString());
            writer.WriteElementString("BatchSize", batchSize.ToString());

            writer.WriteStartElement("Layers");
            foreach (var layer in featureLayers)
            {
                writer.WriteStartElement("Layer");
                writer.WriteElementString("LayerName", "ConvolutionLayer");
                writer.WriteElementString("KernelSize", layer.KernelSize.ToString());
                writer.WriteElementString("Depth", layer.Depth.ToString());
                writer.WriteElementString("ActivationFunction", layer.ActivationFunction.ToString());
                writer.WriteEndElement();
            }
            foreach (var layer in fullyConnectedLayers)
            {
                writer.WriteStartElement("Layer");
                writer.WriteElementString("LayerName", "FullyConnectedLayer");
                writer.WriteElementString("LayerSize", layer.LayerSize.ToString());
                writer.WriteElementString("ActivationFunction", layer.ActivationFunction.ToString());
                writer.WriteEndElement();
            }
            writer.WriteEndElement();

            writer.WriteEndElement();

            FilesCreatorHelper.CloseXmlFile(writer);
        }

        this.learningRate = learningRate;

        for (int epoch = 0; epoch < epochAmount; epoch++)
        {
            data = data.OrderBy(x => random.Next()).ToArray();
            int batchBeginIndex = 0;

            double epochErrorSum = 0.0;

            while (batchBeginIndex < data.Length)
            {
                var batchSamples = batchBeginIndex + batchSize < data.Length ? data.Skip(batchBeginIndex).Take(batchSize) : data[batchBeginIndex..];

                Matrix[] inputSamples = batchSamples.Select(x => x.input).ToArray()!;
                Matrix[] expectsOutputs = batchSamples.Select(x => x.output).ToArray()!;

                double batchErrorSum = 0.0;

                if (multiThreaded)
                {
                    Parallel.ForEach(batchSamples, (sample) =>
                    {
                        (Matrix prediction, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation) = Feedforward(sample.input);
                        Backpropagation(sample.output, prediction, featureLayersOutputs, fullyConnectedLayersOutputBeforeActivation);

                        batchErrorSum += Utilities.CalculateCrossEntropyCost(sample.output, prediction);
                    });
                }
                else
                {
                    foreach (var sample in batchSamples)
                    {
                        (Matrix prediction, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation) = Feedforward(sample.input);
                        Backpropagation(sample.output, prediction, featureLayersOutputs, fullyConnectedLayersOutputBeforeActivation);

                        batchErrorSum += Utilities.CalculateCrossEntropyCost(sample.output, prediction);
                    }
                }


                foreach (var layer in fullyConnectedLayers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }
                foreach (var layer in featureLayers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }

                double epochPercentFinish = 100 * batchBeginIndex / (double)data.Length;
                Console.WriteLine($"Epoch: {epoch + 1}\n" +
                                    $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                                    $"Batch error: {batchErrorSum / batchSize}\n");

                epochErrorSum += batchErrorSum;

                batchBeginIndex += batchSize;
            }

            Console.WriteLine($"\n========================================================\n");
            Console.WriteLine($" [Epoch {epoch + 1} error mean: {epochErrorSum / data.Length}]");
            Console.WriteLine($"\n========================================================\n");

            if (logToFile)
            {
                Console.WriteLine("Calculating correct predictions...");
                int correctPredictions = 0;
                Parallel.ForEach(data, (sample) =>
                {
                    (Matrix prediction, _, _) = Feedforward(sample.input);
                    if (prediction.IndexOfMax() == sample.output.IndexOfMax())
                        correctPredictions++;
                });
                Console.WriteLine("Correct predictions calculated: " + correctPredictions * 100.0 / data.Length + "\n");

                XDocument xml = XDocument.Load(logFilePath + "_" + guid.ToString() + ".xml");
                var root = xml.Root!;
                var learningIte = root.Elements("LearningInstance").First();
                learningIte!.Add(
                    new XElement("Epoch",
                        new XElement("EpochNumber", epoch + 1),
                        new XElement("ErrorMean", epochErrorSum / data.Length),
                        new XElement("CorrectPredictions", correctPredictions * 100.0 / data.Length)
                    )
                );
                xml.Save(logFilePath + "_" + guid.ToString() + ".xml");
            }
        }
    }

    private void LogLearningToFile(string path, bool createHead = false)
    {
        if (createHead)
        {
        }
    }

    public Matrix Predict(Matrix input)
    {
        (Matrix prediction, _, _) = Feedforward(input);
        return prediction;
    }

    public (Matrix output, Matrix[][] featureLayersOutputsBeforeActivation, Matrix[] fullyConnectedLayersOutputBeforeActivation) Feedforward(Matrix input)
    {
        List<Matrix> fullyConnectedLayersOutputBeforeActivation = new List<Matrix>(this.fullyConnectedLayers.Length + 1);
        List<Matrix[]> featureLayersOutputs = new(this.featureLayers.Length + 1);

        Matrix[] currentInput = [input];
        featureLayersOutputs.Add(currentInput);

        for (int i = 0; i < featureLayers.Length; i++)
        {
            (currentInput, var featureOutputBeforeActivation) = featureLayers[i].Forward(currentInput);
            featureLayersOutputs.Add(featureOutputBeforeActivation);

            if (saveFeatureLayersOutputs)
            {
                for (int j = 0; j < currentInput.Length; j++)
                {
                    ImagesProcessor.DataReader.SaveToImage(currentInput[j].ToArray(), $"./../../../featureLayer_{i}_{j}.png");
                }
            }
        }

        var flattenedMatrix = Utilities.FlattenMatrix(currentInput);
        fullyConnectedLayersOutputBeforeActivation.Add(flattenedMatrix);

        (Matrix activatedOutput, Matrix outputBeforeActivation) = fullyConnectedLayers[0].Forward(flattenedMatrix, flattenedMatrix);
        fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);

        for (int i = 1; i < fullyConnectedLayers.Length; i++)
        {
            (activatedOutput, outputBeforeActivation) = fullyConnectedLayers[i].Forward(activatedOutput, outputBeforeActivation);
            fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);
        }

        return (activatedOutput, featureLayersOutputs.ToArray(), fullyConnectedLayersOutputBeforeActivation.ToArray());
    }

    public void Backpropagation(Matrix expectedResult, Matrix prediction, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);

        for (int i = fullyConnectedLayers.Length - 1; i >= 0; i--)
        {
            error = fullyConnectedLayers[i].Backward(error, fullyConnectedLayersOutputBeforeActivation[i], fullyConnectedLayersOutputBeforeActivation[i + 1], learningRate);
        }

        Matrix[] errorMatrices = Utilities.UnflattenMatrix(error, outputFromLastFeatureLayerSize.rows);

        for (int i = featureLayers.Length - 1; i >= 0; i--)
        {
            var thisLayerOutBeforeActivation = featureLayersOutputs[i+1];
            var prevLayerOutBeforeActivation = featureLayersOutputs[i];
            errorMatrices = featureLayers[i].Backward(errorMatrices, prevLayerOutBeforeActivation, thisLayerOutBeforeActivation, learningRate);
        }
    }
}