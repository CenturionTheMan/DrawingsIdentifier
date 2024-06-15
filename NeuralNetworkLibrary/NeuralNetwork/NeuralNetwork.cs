using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using MyBaseLibrary;

namespace NeuralNetworkLibrary;

public class NeuralNetwork : INeuralNetwork
{
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

    private Action<int, int, double>? onLearningIteration; //epoch, sample index, error
    private Action<int, float, double>? onBatchLearningIteration; //epoch, epochPercentFinish, error(mean)

    private static Random random = new Random();

    private ILayer[] layers;
 
    internal double LearningRate;
    private (int rows, int columns) outputFromLastFeatureLayerSize;

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
                    if(isPrevFullyConnected)
                    {
                        throw new InvalidOperationException("Convolution layer should be after another convolution layer or pooling layer");
                    }

                    var layer = new ConvolutionLayer(currentInput, currentTemplate.KernelSize, currentTemplate.Depth, currentTemplate.Stride, currentTemplate.ActivationFunction, currentTemplate.MinWeight, currentTemplate.MaxWeight);
                    layers.Add(layer);
                    var size = MatrixExtender.GetSizeAfterConvolution((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), (currentTemplate.KernelSize, currentTemplate.KernelSize), currentTemplate.Stride);
                    currentInput = (currentTemplate.Depth, size.outputRows, size.outputColumns);
                    break;

                case LayerType.Pooling:
                    if(isPrevFullyConnected)
                    {
                        throw new InvalidOperationException("Pooling layer should be after another convolution layer or pooling layer");
                    }

                    var poolingLayer = new PoolingLayer(currentTemplate.PoolSize, currentTemplate.Stride);
                    layers.Add(poolingLayer);
                    var poolingSize = MatrixExtender.GetSizeAfterPooling((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), currentTemplate.PoolSize, currentTemplate.Stride);                    
                    currentInput = (currentInput.inputDepth, poolingSize.outputRows, poolingSize.outputColumns);
                    break;

                case LayerType.FullyConnected:
                    if(!isPrevFullyConnected)
                    {
                        var reshapeLayer = ReshapeInput(true, currentInput.inputRowsAmount, currentInput.inputColumnsAmount);
                        currentInput = (1, currentInput.inputRowsAmount * currentInput.inputColumnsAmount * currentInput.inputDepth, 1);
                        layers.Add(reshapeLayer);
                    }

                    var fullyConnectedLayer = new FullyConnectedLayer(currentInput.inputRowsAmount, currentTemplate.LayerSize, currentTemplate.ActivationFunction, currentTemplate.MinWeight, currentTemplate.MaxWeight);
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

    public Task TrainOnNewTask((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken=default)
    {
        return Task.Run(() => Train(data, learningRate, epochAmount, batchSize, cancellationToken), cancellationToken);
    }

    public void Train((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken=default)
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

                    if(double.IsNaN(error) || double.IsInfinity(error) || double.IsNegativeInfinity(error))
                    {
                        throw new InvalidOperationException("Error is NaN or Infinity");
                    }

                    Backpropagation(batchSamples[i].output, prediction, outputsBeforeActivation);

                    
                    OnLearningIteration?.Invoke(epoch, batchBeginIndex+i, error);
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
        }
    }

    public Matrix Predict(Matrix input)
    {
        Matrix[] currentInput = [input];

        for (int i = 0; i < layers.Length; i++)
        {
            (currentInput, _) = layers[i].Forward(currentInput);
        }

        if(currentInput.Length != 1)
            throw new InvalidOperationException("Prediction should return only one matrix");
        return currentInput[0];
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

    internal (Matrix output, Matrix[][] layersBeforeActivation) Feedforward(Matrix input)
    {
        List<Matrix[]> layersBeforeActivation = new(this.layers.Length + 1);

        Matrix[] currentInput = [input];
        layersBeforeActivation.Add(currentInput);

        for (int i = 0; i < layers.Length; i++)
        {
            (currentInput, var otherOutput) = layers[i].Forward(currentInput);

            layersBeforeActivation.Add(otherOutput);
        }

        if(currentInput.Length != 1)
            throw new InvalidOperationException("Prediction should return only one matrix");

        return (currentInput[0], layersBeforeActivation.ToArray());
    }

    internal void Backpropagation(Matrix expectedResult, Matrix prediction, Matrix[][] layersBeforeActivation)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);

        Matrix[] currentError = [error];

        for (int i = layers.Length - 1; i >= 0; i--)
        {
            var thisLayerOutBeforeActivation = layersBeforeActivation[i+1];
            var prevLayerOutBeforeActivation = layersBeforeActivation[i];
            currentError = layers[i].Backward(currentError, prevLayerOutBeforeActivation, thisLayerOutBeforeActivation, LearningRate);
        }
    }
}