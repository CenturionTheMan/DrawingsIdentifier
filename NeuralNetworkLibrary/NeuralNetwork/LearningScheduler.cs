namespace NeuralNetworkLibrary;

public class LearningScheduler
{
    internal CancellationTokenSource cts;

    internal double initialLearningRate;
    internal int epochAmount;
    internal int batchSize;


    int epochDropCount;

    public LearningScheduler(double initialLearningRate, int epochAmount, int batchSize, int epochDropCount)
    {
        this.initialLearningRate = initialLearningRate;
        this.epochDropCount = epochDropCount;
        this.epochAmount = epochAmount;
        this.batchSize = batchSize;
        cts = new CancellationTokenSource();
    }


    internal void SetLearningNanny(NeuralNetwork neuralNetwork)
    {
        double learningRate = this.initialLearningRate*2;

        neuralNetwork.OnEpochLearningIteration += (epoch, correctness) =>
        {
            if (epoch % this.epochDropCount == 0)
            {
                learningRate /= 2;
                neuralNetwork.LearningRate = learningRate;
            }
        };

        Queue<double> errors = new Queue<double>(3);
        neuralNetwork.OnBatchLearningIteration += (epoch, epochPercentFinish, error) =>
        {
            if(double.IsNaN(error))
            {
                cts.Cancel();
                return;
            }
            if(errors.Count == 0)
            {
                errors.Enqueue(error);
                return;
            }

            var avg = errors.Average();

            if (error > avg * 1.5)
            {
                learningRate /= 2;
                neuralNetwork.LearningRate = learningRate;
            }   

            errors.Enqueue(error);

            if (errors.Count > 3)
            {
                errors.Dequeue();
            }
        };
    }
}