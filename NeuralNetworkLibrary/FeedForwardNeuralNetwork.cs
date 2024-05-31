using System.Diagnostics;

namespace NeuralNetworkLibrary;

/// <summary>
/// Represents a Convolutional Neural Network.
/// </summary>
/// <remarks>
/// This class provides functionality for training and predicting using a Convolutional Neural Network.
/// It supports various activation functions and implements backpropagation for learning.
/// </remarks>
public class FeedForwardNeuralNetwork : INeuralNetwork
{
    private static Random random = new Random();

    private int[] layersSizes;
    private Matrix[] biasesForLayers;
    private Matrix[] weightsForLayers;
    private Matrix[] layersBeforeActivation;
    private double learningRate;
    private ActivationFunction[] activationFunctions;
    private int layersAmount;

    private Action<int, double, double>? onLearningIteration;
    public Action<int, double, double>? OnLearningIteration { get => onLearningIteration; set => onLearningIteration = value; }

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="FeedForwardNeuralNetwork"/> class.
    /// </summary>
    /// <param name="layersSizes"></param>
    /// <param name="activationFunctions"></param>
    /// <exception cref="ArgumentException"></exception>
    public FeedForwardNeuralNetwork(int[] layersSizes, ActivationFunction[] activationFunctions)
    {
        if (layersSizes.Length != activationFunctions.Length + 1)
        {
            throw new ArgumentException("Activation functions amount must be equal to layers amount - 1");
        }

        this.layersSizes = layersSizes;
        this.layersAmount = layersSizes.Length;

        this.activationFunctions = activationFunctions;

        this.layersBeforeActivation = new Matrix[layersAmount];
        this.biasesForLayers = new Matrix[layersAmount - 1];
        this.weightsForLayers = new Matrix[layersAmount - 1];

        for (int i = 0; i < layersAmount; i++)
        {
            this.layersBeforeActivation[i] = new Matrix(layersSizes[i], 1);
        }

        for (int i = 0; i < layersAmount - 1; i++)
        {
            int fromLayer = layersSizes[i];
            int toLayer = layersSizes[i + 1];
            this.weightsForLayers[i] = new Matrix(toLayer, fromLayer, -0.25, 0.25);
            this.biasesForLayers[i] = new Matrix(toLayer, 1, -0.1, 0.1);
        }
    }

    #endregion Constructors

    #region Training and Predicting

    /// <summary>
    /// Trains the network using the given data.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="learningRate"></param>
    /// <param name="epochAmount"></param>
    /// <param name="batchSize"></param>
    /// <param name="expectedMaxError"></param>
    /// <param name="onIteration"></param>
    /// <exception cref="ArgumentException"></exception>
    public void Train((double[] inputs, double[] outputs)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001)
    {
        if (data[0].inputs.Length != layersSizes[0])
        {
            throw new ArgumentException("Inputs length must be equal to the first layer size");
        }

        if (data[0].outputs.Length != layersSizes[layersAmount - 1])
        {
            throw new ArgumentException("Outputs length must be equal to the last layer size");
        }

        this.learningRate = learningRate;

        for (int epoch = 0; epoch < epochAmount; epoch++)
        {
            data = data.OrderBy(x => random.Next()).ToArray();
            int batchBeginIndex = 0;

            while (batchBeginIndex < data.Length)
            {
                var batchSamples = batchBeginIndex + batchSize < data.Length ? data.Skip(batchBeginIndex).Take(batchSize) : data[batchBeginIndex..];

                Matrix[] inputSamples = batchSamples.Select(x => new Matrix(x.inputs)).ToArray()!;
                Matrix[] expectsOutputsSamples = batchSamples.Select(x => new Matrix(x.outputs)).ToArray()!;

                double batchError = PerformLearningIteration(inputSamples, expectsOutputsSamples);

                onLearningIteration?.Invoke(epoch + 1, 100 * batchBeginIndex / (double)data.Length, batchError);

                if (batchError < expectedMaxError)
                {
                    return;
                }

                batchBeginIndex += batchSize;
            }
        }
    }

    /// <summary>
    /// Predicts the output for the given inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public double[] Predict(double[] inputs)
    {
        if (inputs.Length != layersSizes[0])
        {
            throw new ArgumentException("Inputs length must be equal to the first layer size");
        }

        var result = Feedforward(new Matrix(inputs));

        List<double> resultList = new List<double>();

        for (int i = 0; i < result.RowsAmount; i++)
        {
            resultList.Add(result.Values[i, 0]);
        }

        return resultList.ToArray();
    }

    #endregion Training and Predicting

    #region Base Methods

    /// <summary>
    /// Performs a learning iteration using the given data samples (mini batch) and expected results.
    /// </summary>
    /// <param name="dataSamples"></param>
    /// <param name="expectedResults"></param>
    /// <returns></returns>
    private double PerformLearningIteration(Matrix[] dataSamples, Matrix[] expectedResults)
    {
        Matrix[] changesForWeightsSum = new Matrix[layersAmount - 1];
        Matrix[] changesForBiasesSum = new Matrix[layersAmount - 1];
        for (int i = 0; i < layersAmount - 1; i++)
        {
            changesForWeightsSum[i] = new Matrix(weightsForLayers[i].RowsAmount, weightsForLayers[i].ColumnsAmount);
            changesForBiasesSum[i] = new Matrix(biasesForLayers[i].RowsAmount, biasesForLayers[i].ColumnsAmount);
        }

        double errorSum = 0.0;

        Parallel.For(0, dataSamples.Length, i =>
        {
            Matrix prediction = Feedforward(dataSamples[i]);

            var changes = Backpropagation(expectedResults[i], prediction);

            errorSum += activationFunctions[^1] == ActivationFunction.Softmax ? CalculateCrossEntropyCost(expectedResults[i], prediction) : CalculateMeanSquaredError(expectedResults[i], prediction);

            for (int j = 0; j < layersAmount - 1; j++)
            {
                changesForWeightsSum[j] = changesForWeightsSum[j].ElementwiseAdd(changes.changeForWeights[j]);
                changesForBiasesSum[j] = changesForBiasesSum[j].ElementwiseAdd(changes.changeForBiases[j]);
            }
        });

        Parallel.For(0, layersAmount - 1, i =>
        {
            weightsForLayers[i] = weightsForLayers[i].ElementwiseAdd(changesForWeightsSum[i].ApplyFunction(x => x / dataSamples.Length));
            biasesForLayers[i] = biasesForLayers[i].ElementwiseAdd(changesForBiasesSum[i].ApplyFunction(x => x / dataSamples.Length));
        });

        return errorSum / (double)dataSamples.Length;
    }

    /// <summary>
    /// Feeds the input through the network and returns the output.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    private Matrix Feedforward(Matrix input)
    {
        layersBeforeActivation[0] = input;
        Matrix currentLayer = layersBeforeActivation[0];

        for (int i = 0; i < layersAmount - 1; i++)
        {
            Matrix multipliedByWeightsLayer = Matrix.DotProductMatrices(weightsForLayers[i], currentLayer);

            Matrix layerWithAddedBiases = multipliedByWeightsLayer.ElementwiseAdd(biasesForLayers[i]);

            Matrix activatedLayer = activationFunctions[i] switch
            {
                ActivationFunction.ReLU => ReLU(layerWithAddedBiases),
                ActivationFunction.Sigmoid => Sigmoid(layerWithAddedBiases),
                ActivationFunction.Softmax => Softmax(layerWithAddedBiases),
                _ => throw new NotImplementedException()
            };

            layersBeforeActivation[i + 1] = layerWithAddedBiases;
            currentLayer = activatedLayer;
        }

        return currentLayer;
    }

    /// <summary>
    /// Performs backpropagation and returns the changes for weights and biases.
    /// </summary>
    /// <param name="expectedResults"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    private (Matrix[] changeForWeights, Matrix[] changeForBiases) Backpropagation(Matrix expectedResults, Matrix predictions)
    {
        Matrix[] changeForWeights = new Matrix[layersAmount - 1];
        Matrix[] changeForBiases = new Matrix[layersAmount - 1];

        Matrix errorMatrix = expectedResults.ElementwiseSubtract(predictions);

        for (int i = layersAmount - 2; i >= 0; i--)
        {
            Matrix activationDerivativeLayer = activationFunctions[i] switch
            {
                ActivationFunction.ReLU => DerivativeReLU(layersBeforeActivation[i + 1]),
                ActivationFunction.Sigmoid => DerivativeSigmoid(layersBeforeActivation[i + 1]),
                ActivationFunction.Softmax => DerivativeSoftmax(layersBeforeActivation[i + 1]),
                _ => throw new NotImplementedException()
            };

            Matrix gradientMatrix = activationDerivativeLayer.ElementwiseMultiply(errorMatrix).ApplyFunction(x => x * learningRate);

            Matrix deltaWeightsMatrix = Matrix.DotProductMatrices(gradientMatrix, layersBeforeActivation[i].Transpose());

            changeForWeights[i] = deltaWeightsMatrix;
            changeForBiases[i] = gradientMatrix;

            errorMatrix = Matrix.DotProductMatrices(weightsForLayers[i].Transpose(), errorMatrix);
        }

        return (changeForWeights, changeForBiases);
    }

    #endregion Base Methods

    #region Activation Functions and Error

    /// <summary>
    /// Calculates the mean squared error between the expected and predicted results.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    private double CalculateMeanSquaredError(Matrix expected, Matrix predictions)
    {
        if (predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        double sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += Math.Pow(expected.Values[i, j] - predictions.Values[i, j], 2);
            }
        }

        return sum / (predictions.RowsAmount * predictions.ColumnsAmount);
    }

    /// <summary>
    /// Calculates the cross entropy cost between the expected and predicted results.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    private double CalculateCrossEntropyCost(Matrix expected, Matrix predictions)
    {
        if (predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        double sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += expected.Values[i, j] * Math.Log(predictions.Values[i, j]);
            }
        }

        return -sum;
    }

    /// <summary>
    /// Applies the ReLU activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    private Matrix ReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x > 0 ? x : 0; });
    }

    /// <summary>
    /// Applies the derivative of the ReLU activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    private Matrix DerivativeReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x >= 0 ? 1.0 : 0.0; });
    }

    /// <summary>
    /// Applies the Sigmoid activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    private Matrix Sigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x => 1 / (1 + Math.Exp(-x)));
    }

    /// <summary>
    /// Applies the derivative of the Sigmoid activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    private Matrix DerivativeSigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x =>
        {
            var sig = 1 / (1 + Math.Exp(-x));
            return sig * (1 - sig);
        });
    }

    /// <summary>
    /// Applies the Softmax activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    private Matrix Softmax(Matrix mat)
    {
        var expMat = mat.ApplyFunction(x => Math.Exp(x));
        double sumOfMatrix = expMat.Sum();
        return expMat.ApplyFunction(x => x / sumOfMatrix);
    }

    /// <summary>
    /// Applies the derivative of the Softmax activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    private Matrix DerivativeSoftmax(Matrix mat)
    {
        return Softmax(mat).ApplyFunction(x => x * (1 - x));
    }

    #endregion Activation Functions and Error

    #region Helper Methods

    public bool IsStructureEqual(int[] layersSize, ActivationFunction[] activationFunctions)
    {
        if (layersSize.Length != layersSizes.Length || activationFunctions.Length != this.activationFunctions.Length)
        {
            return false;
        }

        for (int i = 0; i < layersSize.Length; i++)
        {
            if (layersSize[i] != layersSizes[i])
            {
                return false;
            }
        }

        for (int i = 0; i < activationFunctions.Length; i++)
        {
            if (activationFunctions[i] != this.activationFunctions[i])
            {
                return false;
            }
        }

        return true;
    }

    #endregion Helper Methods
}

public enum ActivationFunction
{
    ReLU,
    Sigmoid,
    Softmax,
}