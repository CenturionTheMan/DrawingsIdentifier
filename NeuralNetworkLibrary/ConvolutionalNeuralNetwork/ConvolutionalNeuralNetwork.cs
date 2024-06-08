using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public class ConvolutionalNeuralNetwork
{
    private static Random random = new Random();


    private IFeatureExtractionLayer[] featureLayers;
    private FullyConnectedLayer[] fullyConnectedLayers;
    private double learningRate;

    public ConvolutionalNeuralNetwork(IFeatureExtractionLayer[] featureExtractionLayers, FullyConnectedLayer[] fullyConnectedLayers)
    {
        this.featureLayers = featureExtractionLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;
    }



    //TODO REDESIGN: stop averaging errors, calculate change for each biases from each sample and then average that change
    public void Train((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001)
    {
        this.learningRate = learningRate;

        for (int epoch = 0; epoch < epochAmount; epoch++)
        {
            data = data.OrderBy(x => random.Next()).ToArray();
            int batchBeginIndex = 0;

            while (batchBeginIndex < data.Length)
            {
                var batchSamples = batchBeginIndex + batchSize < data.Length ? data.Skip(batchBeginIndex).Take(batchSize) : data[batchBeginIndex..];

                Matrix[] inputSamples = batchSamples.Select(x => x.input).ToArray()!;
                Matrix[] expectsOutputs= batchSamples.Select(x => x.output).ToArray()!;

                double batchErrorSum = 0.0;


                Parallel.ForEach(batchSamples, (sample) =>
                {
                    (Matrix prediction, Matrix[][] featureLayersOutputs,Matrix[] fullyConnectedLayersOutputBeforeActivation, IEnumerable<(int, int)> dimensions) = Feedforward(sample.input);
                    Backpropagation(sample.output, prediction, featureLayersOutputs, fullyConnectedLayersOutputBeforeActivation, dimensions);

                    batchErrorSum += Utilities.CalculateCrossEntropyCost(sample.output, prediction);
                });
                
                foreach (var layer in fullyConnectedLayers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }
                foreach (var layer in featureLayers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }

                // Parallel.ForEach(fullyConnectedLayers, (layer) =>
                // {
                //     layer.UpdateWeightsAndBiases(batchSize);
                // });

                // Parallel.ForEach(featureLayers, (layer) =>
                // {
                //     layer.UpdateWeightsAndBiases(batchSize);
                // });
                    
                double epochPercentFinish = 100 * batchBeginIndex / (double)data.Length;
                 Console.WriteLine( $"Epoch: {epoch + 1}\n" +
                                    $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                                    $"Batch error: {batchErrorSum / batchSize}\n");

                // onLearningIteration?.Invoke(epoch + 1, 100 * batchBeginIndex / (double)data.Length, batchError);
                // if (batchError < expectedMaxError)
                // {
                //     return;
                // }


                batchBeginIndex += batchSize;
            }
        }
    }

    public Matrix Predict(Matrix input)
    {
        (Matrix prediction, _, _, _) = Feedforward(input);
        return prediction;
    }

    public (Matrix output, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation, IEnumerable<(int, int)> convolutionMatricesDimensions) Feedforward(Matrix input)
    {
        List<Matrix> fullyConnectedLayersOutputBeforeActivation = new List<Matrix>(this.fullyConnectedLayers.Length + 1);
        List<Matrix[]> featureLayersOutputs = new (this.featureLayers.Length + 1);

        Matrix[] currentInput = [input];
        featureLayersOutputs.Add(currentInput);

        for (int i = 0; i < featureLayers.Length; i++)
        {
            currentInput = featureLayers[i].Forward(currentInput);
            featureLayersOutputs.Add(currentInput);
        }

        (var flattenedMatrix, var dimensions) = Utilities.FlattenMatrices(currentInput);
        fullyConnectedLayersOutputBeforeActivation.Add(flattenedMatrix);


        if(fullyConnectedLayers[0].IsInitialized == false)
            fullyConnectedLayers[0].InitializeWeightsAndBiases(flattenedMatrix.RowsAmount, -0.2, 0.2);
        (Matrix activatedOutput, Matrix outputBeforeActivation) = fullyConnectedLayers[0].Forward(flattenedMatrix, flattenedMatrix);
        fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);


        for (int i = 1; i < fullyConnectedLayers.Length; i++)
        {
            if(fullyConnectedLayers[i].IsInitialized == false)
                fullyConnectedLayers[i].InitializeWeightsAndBiases(fullyConnectedLayers[i - 1].LayerSize, -0.2, 0.2);
            
            (activatedOutput, outputBeforeActivation) = fullyConnectedLayers[i].Forward(activatedOutput, outputBeforeActivation);
            fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);
        }

        return (activatedOutput, featureLayersOutputs.ToArray(), fullyConnectedLayersOutputBeforeActivation.ToArray(), dimensions);
    }

    public void Backpropagation(Matrix expectedResult, Matrix prediction, Matrix[][] featureLayersOutputs, Matrix[] fullyConnectedLayersOutputBeforeActivation, IEnumerable<(int, int)> convolutionMatricesDimensions)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);
        
        for (int i = fullyConnectedLayers.Length - 1; i >= 0; i--)
        {
            error = fullyConnectedLayers[i].Backward(error, fullyConnectedLayersOutputBeforeActivation[i], fullyConnectedLayersOutputBeforeActivation[i + 1], learningRate);
        }

        Matrix[] errorMatrices = Utilities.RecreateMatrices(error, convolutionMatricesDimensions);

        for (int i = featureLayers.Length - 1; i >= 0; i--)
        {
            errorMatrices = featureLayers[i].Backward(errorMatrices, featureLayersOutputs[i+1], learningRate);
        }
    }

}