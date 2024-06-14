using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using MyBaseLibrary;

namespace NeuralNetworkLibrary;

public class ConvolutionalNeuralNetwork : INeuralNetwork
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

    private IFeatureExtractionLayer[] featureLayers;
    private FullyConnectedLayer[] fullyConnectedLayers;
 
    internal double LearningRate;
    private (int rows, int columns) outputFromLastFeatureLayerSize;


    public ConvolutionalNeuralNetwork((int depth, int rows, int columns) input, IFeatureExtractionLayer[] featureExtractionLayers, FullyConnectedLayer[] fullyConnectedLayers)
    {
        this.featureLayers = featureExtractionLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;

        (int nextDepth, int nextHeight, int nextWidth) = input;
        for (int i = 0; i < featureExtractionLayers.Length; i++)
        {
            (nextDepth, nextHeight, nextWidth) = featureExtractionLayers[i].Initialize((nextDepth, nextHeight, nextWidth));
        }

        outputFromLastFeatureLayerSize = (nextHeight, nextWidth);

        if (fullyConnectedLayers.Length > 0)
        {
            int depth = featureExtractionLayers.Length > 0 ? nextDepth : input.depth;
            fullyConnectedLayers[0].Initialize(outputFromLastFeatureLayerSize.rows * outputFromLastFeatureLayerSize.columns * depth);
            for (int i = 1; i < fullyConnectedLayers.Length; i++)
            {
                fullyConnectedLayers[i].Initialize(fullyConnectedLayers[i - 1].LayerSize);
            }
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

                    (Matrix prediction, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation) = Feedforward(batchSamples[i].input);
                    prediction = prediction + double.Epsilon;
                    
                    Backpropagation(batchSamples[i].output, prediction, featureLayersOutputs, fullyConnectedLayersOutputBeforeActivation);

                    double error = ActivationFunctionsHandler.CalculateCrossEntropyCost(batchSamples[i].output, prediction);
                    batchErrorSum += error;
                    OnLearningIteration?.Invoke(epoch, batchBeginIndex+i, error);
                });

                if (cancellationToken.IsCancellationRequested) return;


                foreach (var layer in fullyConnectedLayers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }
                foreach (var layer in featureLayers)
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

        for (int i = 0; i < featureLayers.Length; i++)
        {
            (currentInput, _) = featureLayers[i].Forward(currentInput);
        }

        var flattenedMatrix = MatrixExtender.FlattenMatrix(currentInput);

        (Matrix activatedOutput, _) = fullyConnectedLayers[0].Forward(flattenedMatrix);

        for (int i = 1; i < fullyConnectedLayers.Length; i++)
        {
            (activatedOutput, _) = fullyConnectedLayers[i].Forward(activatedOutput);
        }

        return activatedOutput;
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

    internal (Matrix output, Matrix[][] featureLayersOutputsBeforeActivation, Matrix[] fullyConnectedLayersOutputBeforeActivation) Feedforward(Matrix input)
    {
        List<Matrix> fullyConnectedLayersOutputBeforeActivation = new List<Matrix>(this.fullyConnectedLayers.Length + 1);
        List<Matrix[]> featureLayersOutputs = new(this.featureLayers.Length + 1);

        Matrix[] currentInput = [input];
        featureLayersOutputs.Add(currentInput);

        for (int i = 0; i < featureLayers.Length; i++)
        {
            (currentInput, var featureOutputBeforeActivation) = featureLayers[i].Forward(currentInput);
            featureLayersOutputs.Add(featureOutputBeforeActivation);
        }

        var flattenedMatrix = MatrixExtender.FlattenMatrix(currentInput);
        fullyConnectedLayersOutputBeforeActivation.Add(flattenedMatrix);

        (Matrix activatedOutput, Matrix outputBeforeActivation) = fullyConnectedLayers[0].Forward(flattenedMatrix);
        fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);

        for (int i = 1; i < fullyConnectedLayers.Length; i++)
        {
            (activatedOutput, outputBeforeActivation) = fullyConnectedLayers[i].Forward(activatedOutput);
            fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);
        }

        return (activatedOutput, featureLayersOutputs.ToArray(), fullyConnectedLayersOutputBeforeActivation.ToArray());
    }

    internal void Backpropagation(Matrix expectedResult, Matrix prediction, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);

        for (int i = fullyConnectedLayers.Length - 1; i >= 0; i--)
        {
            error = fullyConnectedLayers[i].Backward(error, fullyConnectedLayersOutputBeforeActivation[i], fullyConnectedLayersOutputBeforeActivation[i + 1], LearningRate);
        }

        Matrix[] errorMatrices = MatrixExtender.UnflattenMatrix(error, outputFromLastFeatureLayerSize.rows);

        for (int i = featureLayers.Length - 1; i >= 0; i--)
        {
            var thisLayerOutBeforeActivation = featureLayersOutputs[i+1];
            var prevLayerOutBeforeActivation = featureLayersOutputs[i];
            errorMatrices = featureLayers[i].Backward(errorMatrices, prevLayerOutBeforeActivation, thisLayerOutBeforeActivation, LearningRate);
        }
    }
}