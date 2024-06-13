using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public interface INeuralNetwork
{
    /// <summary>
    /// Event that is called after each learning iteration
    /// augments: epoch, sample index, error
    /// </summary>
    public Action<int, int, double>? OnLearningIteration
    {
        get;
        set;
    }

    /// <summary>
    /// Event that is called after each batch learning iteration
    /// augments: epoch, epochPercentFinish, error(mean)
    /// </summary>
    public Action<int, float, double>? OnBatchLearningIteration
    {
        get;
        set;
    }

    /// <summary>
    /// Trains the neural network on the given data
    /// Learning will be done on a separate task
    /// </summary>
    /// <param name="data">Data to train on</param>
    /// <param name="learningRate">Learning rate</param>
    /// <param name="epochAmount">Amount of epochs</param>
    /// <param name="batchSize">Size of the batch</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task</returns>
    public Task TrainOnNewTask((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken=default);
    
    /// <summary>
    /// Trains the neural network on the given data
    /// Learning will be done on current thread
    /// </summary>
    /// <param name="data">Data to train on</param>
    /// <param name="learningRate">Learning rate</param>
    /// <param name="epochAmount">Amount of epochs</param>
    /// <param name="batchSize">Size of the batch</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public void Train((Matrix input, Matrix output)[] data, double learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken=default);

    /// <summary>
    /// Predicts the output for the given input
    /// </summary>
    /// <param name="input">Input</param>
    /// <returns>Predicted output</returns>
    public Matrix Predict(Matrix input);

    /// <summary>
    /// Calculates the correctness of the neural network
    /// </summary>
    /// <param name="testData">Data to test on</param>
    /// <returns>Correctness in percentage</returns>
    public float CalculateCorrectness((Matrix input, Matrix expectedOutput)[] testData);
}