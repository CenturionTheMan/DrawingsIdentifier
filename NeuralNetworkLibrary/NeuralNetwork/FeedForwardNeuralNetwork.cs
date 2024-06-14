namespace NeuralNetworkLibrary;

public class FeedForwardNeuralNetwork : INeuralNetwork
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

    internal double LearningRate;


    private FullyConnectedLayer[] fullyConnectedLayers;

    public FeedForwardNeuralNetwork(int inputSize, FullyConnectedLayer[] fullyConnectedLayers)
    {
        this.fullyConnectedLayers = fullyConnectedLayers;

        if (fullyConnectedLayers.Length > 0)
        {
            fullyConnectedLayers[0].Initialize(inputSize);
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

        data = data.Select(x => (MatrixExtender.FlattenMatrix(x.input), x.output)).ToArray();

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

                    (Matrix prediction, Matrix[] fullyConnectedLayersOutputBeforeActivation) = Feedforward(batchSamples[i].input);
                    prediction = prediction + double.Epsilon;

                    Backpropagation(batchSamples[i].output, prediction, fullyConnectedLayersOutputBeforeActivation);

                    double error = ActivationFunctionsHandler.CalculateCrossEntropyCost(batchSamples[i].output, prediction);
                    batchErrorSum += error;
                    OnLearningIteration?.Invoke(epoch, batchBeginIndex+i, error);
                });
                if (cancellationToken.IsCancellationRequested) return;

                foreach (var layer in fullyConnectedLayers)
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
        var (activatedOutput, _) = Feedforward(input);
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

    internal (Matrix output, Matrix[] fullyConnectedLayersOutputBeforeActivation) Feedforward(Matrix input)
    {
        List<Matrix> fullyConnectedLayersOutputBeforeActivation = new List<Matrix>(this.fullyConnectedLayers.Length + 1)
        {
            input
        };

        (Matrix activatedOutput, Matrix outputBeforeActivation) = fullyConnectedLayers[0].Forward(input);
        fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);

        for (int i = 1; i < fullyConnectedLayers.Length; i++)
        {
            (activatedOutput, outputBeforeActivation) = fullyConnectedLayers[i].Forward(activatedOutput);
            fullyConnectedLayersOutputBeforeActivation.Add(outputBeforeActivation);
        }

        return (activatedOutput, fullyConnectedLayersOutputBeforeActivation.ToArray());
    }

    internal void Backpropagation(Matrix expectedResult, Matrix prediction, Matrix[] fullyConnectedLayersOutputBeforeActivation)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);

        for (int i = fullyConnectedLayers.Length - 1; i >= 0; i--)
        {
            error = fullyConnectedLayers[i].Backward(error, fullyConnectedLayersOutputBeforeActivation[i], fullyConnectedLayersOutputBeforeActivation[i + 1], LearningRate);
        }
    }

}