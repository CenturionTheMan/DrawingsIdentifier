using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public class ConvolutionalNeuralNetwork
{
    private const bool multiThreaded = true;

    private static Random random = new Random();


    private IFeatureExtractionLayer[] featureLayers;
    private FullyConnectedLayer[] fullyConnectedLayers;
    private double learningRate;

    //TODO: Change this to be dynamic
    private int kernelSize = 24;

    public ConvolutionalNeuralNetwork(IFeatureExtractionLayer[] featureExtractionLayers, FullyConnectedLayer[] fullyConnectedLayers)
    {
        this.featureLayers = featureExtractionLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;
    }



    public void Train((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001)
    {
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
                Matrix[] expectsOutputs= batchSamples.Select(x => x.output).ToArray()!;

                double batchErrorSum = 0.0;
                

                if(multiThreaded)
                {
                    Parallel.ForEach(batchSamples, (sample) =>
                    {
                        (Matrix prediction, Matrix[][] featureLayersOutputs,Matrix[] fullyConnectedLayersOutputBeforeActivation) = Feedforward(sample.input);
                        Backpropagation(sample.output, prediction, featureLayersOutputs, fullyConnectedLayersOutputBeforeActivation);

                        batchErrorSum += Utilities.CalculateCrossEntropyCost(sample.output, prediction);
                    });
                }
                else
                {
                    foreach (var sample in batchSamples)
                    {
                        (Matrix prediction, Matrix[][] featureLayersOutputs,Matrix[] fullyConnectedLayersOutputBeforeActivation) = Feedforward(sample.input);
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
                Console.WriteLine( $"Epoch: {epoch + 1}\n" +
                                    $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                                    $"Batch error: {batchErrorSum / batchSize}\n");

                epochErrorSum += batchErrorSum;
                // onLearningIteration?.Invoke(epoch + 1, 100 * batchBeginIndex / (double)data.Length, batchError);
                // if (batchError < expectedMaxError)
                // {
                //     return;
                // }


                batchBeginIndex += batchSize;
            }

            Console.WriteLine($"\n========================================================\n");
            Console.WriteLine($" [Epoch {epoch + 1} error mean: {epochErrorSum / data.Length}]");
            Console.WriteLine($"\n========================================================\n");
        }
    }

    public Matrix Predict(Matrix input)
    {
        (Matrix prediction, _, _) = Feedforward(input);
        return prediction;
    }

    public (Matrix output, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation) Feedforward(Matrix input)
    {
        List<Matrix> fullyConnectedLayersOutputBeforeActivation = new List<Matrix>(this.fullyConnectedLayers.Length + 1);
        List<Matrix[]> featureLayersOutputs = new (this.featureLayers.Length + 1);

        Matrix[] currentInput = [input];
        featureLayersOutputs.Add(currentInput);

        for (int i = 0; i < featureLayers.Length; i++)
        {
            (currentInput, var featureOutputBeforeActivation) = featureLayers[i].Forward(currentInput);
            featureLayersOutputs.Add(featureOutputBeforeActivation);
        }

        //TODO uncomment this
        var flattenedMatrix =  Utilities.FlattenMatrix(currentInput);
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

        Matrix[] errorMatrices = Utilities.UnflattenMatrix(error, kernelSize);

        for (int i = featureLayers.Length - 1; i >= 0; i--)
        {
            var prevLayer = featureLayersOutputs[i];
            errorMatrices = featureLayers[i].Backward(errorMatrices, prevLayer, learningRate);
        }
    }

}