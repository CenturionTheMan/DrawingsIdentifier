﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using MyBaseLibrary;

namespace NeuralNetworkLibrary;

public class NeuralNetwork : INeuralNetwork
{
    #region PARAMS

    public Action<int, int, double>? OnLearningIteration
    {
        get => onLearningIteration;
        set => onLearningIteration = value;
    }

    public Action<int, float, double>? OnBatchLearningIteration
    {
        get => onBatchLearningIteration;
        set => onBatchLearningIteration = value;
    }

    public Action<int, float>? OnEpochLearningIteration
    {
        get => onEpochLearningIteration;
        set => onEpochLearningIteration = value;
    }

    public double LearningRate { get; internal set; }
    public float LastTrainCorrectness { get; internal set; }

    private Action<int, int, double>? onLearningIteration; //epoch, sample index, error
    private Action<int, float, double>? onBatchLearningIteration; //epoch, epochPercentFinish, error(mean)
    private Action<int, float>? onEpochLearningIteration; //epoch, correctness

    private static Random random = new Random();

    private ILayer[] layers;

    #endregion PARAMS

    #region CTORS

    private NeuralNetwork(ILayer[] layers, double learningRate, float lastTrainCorrectness)
    {
        this.layers = layers;
        this.LearningRate = learningRate;
        this.LastTrainCorrectness = lastTrainCorrectness;
    }

    public NeuralNetwork(int inputSize, LayerTemplate[] layerTemplates) : this(1, inputSize, 1, layerTemplates)
    {
    }

    public NeuralNetwork(int inputDepth, int inputRowsAmount, int inputColumnsAmount, LayerTemplate[] layerTemplates)
    {
        bool isPrevFullyConnected = false;

        List<ILayer> layers = new List<ILayer>();
        var currentInput = (inputDepth, inputRowsAmount, inputColumnsAmount);

        for (int i = 0; i < layerTemplates.Length; i++)
        {
            var currentTemplate = layerTemplates[i];
            switch (currentTemplate.LayerType)
            {
                case LayerType.Convolution:
                    if (isPrevFullyConnected)
                    {
                        throw new InvalidOperationException("Convolution layer should be after another convolution layer or pooling layer");
                    }

                    var layer = new ConvolutionLayer(currentInput, currentTemplate.KernelSize, currentTemplate.Depth, currentTemplate.Stride, currentTemplate.ActivationFunction);
                    layers.Add(layer);
                    var size = MatrixExtender.GetSizeAfterConvolution((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), (currentTemplate.KernelSize, currentTemplate.KernelSize), currentTemplate.Stride);
                    currentInput = (currentTemplate.Depth, size.outputRows, size.outputColumns);
                    break;

                case LayerType.Pooling:
                    if (isPrevFullyConnected)
                    {
                        throw new InvalidOperationException("Pooling layer should be after another convolution layer or pooling layer");
                    }

                    var poolingLayer = new PoolingLayer(currentTemplate.PoolSize, currentTemplate.Stride);
                    layers.Add(poolingLayer);
                    var poolingSize = MatrixExtender.GetSizeAfterPooling((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), currentTemplate.PoolSize, currentTemplate.Stride);
                    currentInput = (currentInput.inputDepth, poolingSize.outputRows, poolingSize.outputColumns);
                    break;

                case LayerType.FullyConnected:
                    if (!isPrevFullyConnected)
                    {
                        var reshapeLayer = ReshapeInput(true, currentInput.inputRowsAmount, currentInput.inputColumnsAmount);
                        currentInput = (1, currentInput.inputRowsAmount * currentInput.inputColumnsAmount * currentInput.inputDepth, 1);
                        layers.Add(reshapeLayer);
                    }

                    var fullyConnectedLayer = new FullyConnectedLayer(currentInput.inputRowsAmount, currentTemplate.LayerSize, currentTemplate.ActivationFunction);
                    layers.Add(fullyConnectedLayer);
                    currentInput = (1, currentTemplate.LayerSize, 1);
                    isPrevFullyConnected = true;
                    break;

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        this.layers = layers.ToArray();
    }

    private ILayer ReshapeInput(bool featureToClassification, int rows, int columns)
    {
        if (featureToClassification)
        {
            var reshape = new ReshapeFeatureToClassificationLayer(rows, columns);
            return reshape;
        }
        else
        {
            throw new NotImplementedException();
        }
    }

    #endregion CTORS

    #region TRAINING

    public void Train((Matrix input, Matrix output)[] data, LearningScheduler learningScheduler)
    {
        learningScheduler.SetLearningNanny(this);
        Train(data, learningScheduler.initialLearningRate, learningScheduler.epochAmount, learningScheduler.batchSize, learningScheduler.cts.Token);
    }

    public Task TrainOnNewTask((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Train(data, learningRate, epochAmount, batchSize, cancellationToken), cancellationToken);
    }

    public void Train((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken = default)
    {
        this.LearningRate = learningRate;

        for (int epoch = 0; epoch < epochAmount; epoch++)
        {
            data = data.OrderBy(x => random.Next()).ToArray();
            int batchBeginIndex = 0;

            while (batchBeginIndex < data.Length)
            {
                var batchSamples = batchBeginIndex + batchSize < data.Length ? data.Skip(batchBeginIndex).Take(batchSize).ToArray() : data[batchBeginIndex..].ToArray();

                double batchErrorSum = 0;

                Parallel.For(0, batchSamples.Length, (i, loopState) =>
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        loopState.Stop();
                        return;
                    }

                    (Matrix prediction, Matrix[][] outputsBeforeActivation) = Feedforward(batchSamples[i].input);
                    prediction = prediction + double.Epsilon;

                    double error = ActivationFunctionsHandler.CalculateCrossEntropyCost(batchSamples[i].output, prediction);
                    batchErrorSum += error;

                    Backpropagation(batchSamples[i].output, prediction, outputsBeforeActivation);

                    OnLearningIteration?.Invoke(epoch, batchBeginIndex + i, error);
                });

                if (cancellationToken.IsCancellationRequested) return;

                foreach (var layer in layers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }

                float epochPercentFinish = 100 * batchBeginIndex / (float)data.Length;
                OnBatchLearningIteration?.Invoke(epoch, epochPercentFinish, batchErrorSum / batchSize);

                batchBeginIndex += batchSize;
            }

            int toTake = data.Length < 1000 ? data.Length : 1000;
            float correctness = CalculateCorrectness(data.Take(toTake).OrderBy(x => random.Next()).ToArray());
            this.LastTrainCorrectness = correctness;
            OnEpochLearningIteration?.Invoke(epoch, correctness);
        }
    }

    #endregion TRAINING

    #region INTERACTIONS

    public Matrix Predict(Matrix input)
    {
        Matrix[] currentInput = [input];

        for (int i = 0; i < layers.Length; i++)
        {
            (currentInput, _) = layers[i].Forward(currentInput);
        }

        if (currentInput.Length != 1)
            throw new InvalidOperationException("Prediction should return only one matrix");
        return currentInput[0];
    }

    public void SaveFeatureMaps(Matrix input, string directoryPath)
    {
        Matrix[] currentInput = [input];

        for (int i = 0; i < layers.Length; i++)
        {
            (currentInput, _) = layers[i].Forward(currentInput);

            if (layers[i].LayerType == LayerType.Convolution || layers[i].LayerType == LayerType.Pooling)
            {
                for (int j = 0; j < currentInput.Length; j++)
                {
                    var featureMap = currentInput[j];
                    ImagesProcessor.DataReader.SaveToImage(featureMap.ToArray(), directoryPath + $"featureMap_{i}_{j}.png");
                }
            }
        }
    }

    public float CalculateCorrectness((Matrix input, Matrix expectedOutput)[] testData)
    {
        int guessed = 0;

        Parallel.ForEach(testData, item =>
        {
            var prediction = Predict(item.input);
            var max = prediction.Max();

            int predictedNumber = prediction.IndexOfMax();
            int expectedNumber = item.expectedOutput.IndexOfMax();

            if (predictedNumber == expectedNumber)
            {
                Interlocked.Increment(ref guessed);
            }
        });

        return guessed * 100.0f / testData.Length;
    }

    #endregion INTERACTIONS

    #region SAVING / LOADING

    public bool SaveToXmlFile(string path)
    {
        var writer = FilesCreatorHelper.CreateXmlFile(path);
        if (writer == null)
            return false;

        writer.WriteStartElement("Root");

        writer.WriteStartElement("Config");
        writer.WriteElementString("LearningRate", LearningRate.ToString());
        writer.WriteElementString("LayersAmount", layers.Length.ToString());
        writer.WriteElementString("LastTrainCorrectness", LastTrainCorrectness.ToString());
        writer.WriteEndElement();

        writer.WriteStartElement("LayersHead");
        foreach (var layer in layers)
        {
            layer.SaveLayerDescription(writer);
        }
        writer.WriteEndElement();

        writer.WriteStartElement("LayersData");
        foreach (var layer in layers)
        {
            layer.SaveLayerData(writer);
        }
        writer.WriteEndElement();

        writer.WriteEndElement();
        writer.CloseXmlFile();

        return true;
    }

    public static NeuralNetwork? LoadFromXmlFile(string path)
    {
        XDocument xml = XDocument.Load(path);
        var root = xml.Root!;

        var config = root.Element("Config");
        if (config == null)
        {
            return null;
        }

        double learningRate = double.Parse(config.Element("LearningRate")!.Value);
        float lastTrainCorrectness = float.Parse(config.Element("LastTrainCorrectness")!.Value);
        int layersAmount = int.Parse(config.Element("LayersAmount")!.Value);

        var layersHead = root.Element("LayersHead")!.Elements();
        var layersData = root.Element("LayersData")!.Elements();

        ILayer[] layers = new ILayer[layersAmount];
        for (int i = 0; i < layersAmount; i++)
        {
            var layerHead = layersHead.ElementAt(i);
            var layerData = layersData.ElementAt(i);

            var layerTypeStr = layerHead.Attribute("LayerType")!.Value;
            LayerType layerType = Enum.Parse<LayerType>(layerTypeStr);

            ILayer? layer = null;
            switch (layerType)
            {
                case LayerType.Convolution:
                    layer = ConvolutionLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.Pooling:
                    layer = PoolingLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.FullyConnected:
                    layer = FullyConnectedLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.Reshape:
                    layer = ReshapeFeatureToClassificationLayer.LoadLayerData(layerHead, layerData);
                    break;

                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (layer == null)
                return null;

            layers[i] = layer;
        }

        return new NeuralNetwork(layers, learningRate, lastTrainCorrectness);
    }

    #endregion SAVING / LOADING

    #region FORWARD / BACKWARD

    internal (Matrix output, Matrix[][] layersBeforeActivation) Feedforward(Matrix input)
    {
        List<Matrix[]> layersBeforeActivation = new(this.layers.Length + 1);

        Matrix[] currentInput = [input];
        layersBeforeActivation.Add(currentInput);

        for (int i = 0; i < layers.Length; i++)
        {
            (currentInput, var otherOutput) = layers[i].Forward(currentInput);

            //TODO Test with and without it (probably better without but more tests needed)
            // if(layers[i].LayerType == LayerType.Reshape)
            // {
            //     (otherOutput, _) = layers[i].Forward(layersBeforeActivation.Last());
            // }

            layersBeforeActivation.Add(otherOutput);
        }

        if (currentInput.Length != 1)
            throw new InvalidOperationException("Prediction should return only one matrix");

        return (currentInput[0], layersBeforeActivation.ToArray());
    }

    internal void Backpropagation(Matrix expectedResult, Matrix prediction, Matrix[][] layersBeforeActivation)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);

        Matrix[] currentError = [error];

        for (int i = layers.Length - 1; i >= 0; i--)
        {
            var thisLayerOutBeforeActivation = layersBeforeActivation[i + 1];
            var prevLayerOutBeforeActivation = layersBeforeActivation[i];
            currentError = layers[i].Backward(currentError, prevLayerOutBeforeActivation, thisLayerOutBeforeActivation, LearningRate);
        }
    }

    #endregion FORWARD / BACKWARD
}