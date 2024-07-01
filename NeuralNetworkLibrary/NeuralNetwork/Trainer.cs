using System.Diagnostics;

namespace NeuralNetworkLibrary;

public class Trainer
{
    public NeuralNetwork NeuralNetwork => neuralNetwork;

    private NeuralNetwork neuralNetwork;
    private (Matrix[] inputChannels, Matrix output)[] data;
    private float initialLearningRate;
    private float minLearningRate;
    private int epochAmount;
    private int batchSize;

    private bool isPatience = false;
    private float userInitialIgnore = 0.0f;
    private float userPatience = 0;
    private float initialIgnore = 10.0f; //percent
    private int patienceAmount = 50;
    private bool hasCrossedIgnoreThreshold = false;
    private Func<float, float> learningRateModifier = (lr) => lr * 0.9f;

    private bool saveToLog = false;
    private bool saveNN = false;
    private string trainingLogDir = "";
    
    private record TrainingIterationData(int epoch, int dataIndex, float error, float learningRate, float elapsedSeconds);


    /// <summary>
    /// Create a new instance of the <see cref="Trainer"/> class.
    /// </summary>
    /// <param name="neuralNetwork">
    /// Neural network to train.
    /// </param>
    /// <param name="data">
    /// Data to train on.
    /// </param>
    /// <param name="initialLearningRate">
    /// Initial learning rate.
    /// </param>
    /// <param name="minLearningRate">
    /// Minimum learning rate. Meeting this value will stop the training with first epoch end.
    /// </param>
    /// <param name="epochAmount">
    /// Maximum amount of epochs to train.
    /// </param>
    /// <param name="batchSize">
    /// Batch size.
    /// </param>
    public Trainer(NeuralNetwork neuralNetwork, (Matrix[] inputChannels, Matrix output)[] data, float initialLearningRate, float minLearningRate, int epochAmount, int batchSize)
    {
        this.neuralNetwork = neuralNetwork;
        this.data = data;
        this.initialLearningRate = initialLearningRate;
        this.minLearningRate = minLearningRate;
        this.epochAmount = epochAmount;
        this.batchSize = batchSize;
    }

    /// <summary>
    /// Set patience for the training.
    /// </summary>
    /// <param name="initialIgnore">
    /// Initial ignore threshold. Value must be in range [0, 0.95].
    /// Learning rate won t be modified until this threshold is crossed.
    /// 0.0 means that learning rate will be modified from the beginning.
    /// 0.95 means that learning rate will be modified when 95% of first epoch is finished.
    /// </param>
    /// <param name="patience">
    /// Patience is the amount of data that will be used to calculate the slope of the error.
    /// If slope is zero or positive, learning rate will be decreased.
    /// Value must be in range [0, 1].
    /// 0.0 means that learning rate may be modified after each batch.
    /// 1.0 means that learning rate may be modified after each epoch.
    /// </param>
    /// <param name="learningRateModifier">
    /// Modifier for the learning rate. If null, default modifier will be used.
    /// Default modifier is: lr => lr * 0.9f
    /// </param>
    /// <returns>
    /// Trainer instance.
    /// </returns>
    public Trainer SetPatience(float initialIgnore, float patience, Func<float, float>? learningRateModifier = null)
    {
        initialIgnore = Math.Clamp(initialIgnore, 0, 0.95f);
        patience = Math.Clamp(patience, 0, 1f);
        
        this.userInitialIgnore = initialIgnore;
        this.userPatience = patience;

        this.initialIgnore = initialIgnore * 100;
        this.patienceAmount = (int)(data.Length * patience / batchSize);
        this.patienceAmount = Math.Max(1, patienceAmount);

        if(learningRateModifier is not null)
            this.learningRateModifier = learningRateModifier;

        isPatience = true;

        return this;
    }

    /// <summary>
    /// Set saving of the training data.
    /// </summary>
    /// <param name="outputDirPath">
    /// Output directory path. If directory does not exist, it will be created. Logs will be saved in subdirectory with current date and time.
    /// </param>
    /// <param name="saveNN">
    /// Flag determining if neural network should be saved.
    /// </param>
    /// <param name="trainingLogDirectory">
    /// Directory where logs will be saved.
    /// </param>
    /// <returns>
    /// Trainer instance.
    /// </returns>
    public Trainer SetLogSaving(string outputDirPath, bool saveNN, out string trainingLogDirectory)
    {
        if(!Directory.Exists(outputDirPath))
            Directory.CreateDirectory(outputDirPath);

        this.trainingLogDir = outputDirPath + DateTime.Now.ToString("yyyy.MM.dd__HH-mm-ss");

        if(!Directory.Exists(trainingLogDir))
            Directory.CreateDirectory(trainingLogDir);
        this.trainingLogDir += "/";

        this.saveNN = saveNN;
        this.saveToLog = true;

        trainingLogDirectory = this.trainingLogDir;
        return this;
    }

    /// <summary>
    /// Run the training.
    /// </summary>
    public void RunTraining()
    {
        var stopwatch = new Stopwatch();

        var cts = new CancellationTokenSource();

        Stack<TrainingIterationData> trainingIterationData = new();
        List<float> trainCorrectness = new List<float>(epochAmount);

        Queue<(float error, float seconds)>? lastBatchErrors = new();


        neuralNetwork.OnTrainingIteration += (epoch, dataIndex, error) =>
        {
            if(saveToLog)
            {
                var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
                trainingIterationData.Push(new TrainingIterationData(epoch, dataIndex, error, neuralNetwork.LearningRate, (float)elapsedSeconds));
            }

            if(float.IsNaN(error))
            {
                cts?.Cancel();
            }
        };

        neuralNetwork.OnBatchTrainingIteration += (epoch, epochPercentFinish, batchAvgError) =>
        {
            if(isPatience)
            {
                lastBatchErrors!.Enqueue((batchAvgError, (float)stopwatch.Elapsed.TotalSeconds));
                HandlePatience(neuralNetwork, lastBatchErrors, epochPercentFinish);
            }
        };

        neuralNetwork.OnEpochTrainingIteration += (epoch, correctness) =>
        {
            trainCorrectness.Add(correctness);

            if(neuralNetwork.LearningRate <= minLearningRate)
            {
                cts?.Cancel();
            }
        };


        bool hasFinished = false;
        neuralNetwork.OnTrainingFinished += () =>
        {
            if(saveToLog)
                SaveTrainingData(trainingLogDir, trainingIterationData.ToArray(), trainCorrectness.ToArray());

            stopwatch.Stop();
            hasFinished = true;
        };

        stopwatch.Start();
        neuralNetwork.TrainOnNewTask(data, initialLearningRate, epochAmount, batchSize, cts.Token);

        while (hasFinished == false)
        {
            if(cts!.IsCancellationRequested)
            {
                Console.WriteLine("Training has been stopped.");
                Thread.Sleep(1000);
                continue;
            }
            var pressedKey = Console.Read();
            if(pressedKey == 'q')
            {
                cts!.Cancel();
            }
        }
    }

    /// <summary>
    /// Handle patience mechanism.
    /// </summary>
    /// <param name="nn">
    /// Neural network.
    /// </param>
    /// <param name="lastAvgBatchErrors">
    /// Last batch errors.
    /// </param>
    /// <param name="epochPercentFinish">
    /// Percent of epoch finish.
    /// </param>
    private void HandlePatience(NeuralNetwork nn, Queue<(float, float)> lastAvgBatchErrors, float epochPercentFinish)
    {
        if(hasCrossedIgnoreThreshold == false || nn.LearningRate <= minLearningRate)
        {
            if(epochPercentFinish >= initialIgnore)
                hasCrossedIgnoreThreshold = true;
            
            return;
        }

        if(lastAvgBatchErrors.Count() < patienceAmount)return;

        (float slope, _) = Statistics.LinearRegression(lastAvgBatchErrors.ToArray());

        if(slope >= 0)
        {
            nn.LearningRate = Math.Max(minLearningRate, learningRateModifier(nn.LearningRate));
        }
        lastAvgBatchErrors!.Clear();
    }

    /// <summary>
    /// Save training data.
    /// </summary>
    /// <param name="dirPath">
    /// Directory path.
    /// </param>
    /// <param name="trainingIterationData">
    /// Data from training iterations.
    /// </param>
    /// <param name="trainEpochCorrectness">
    /// Epoch correctness.
    /// </param>
    private void SaveTrainingData(string dirPath, TrainingIterationData[] trainingIterationData, float[] trainEpochCorrectness)
    {
        trainingIterationData = trainingIterationData.Where(x => x is not null).ToArray();

        List<object[]> data = [["Epoch", "DataIndex", "Error", "LearningRate", "ElapsedSeconds"]];
        foreach (var item in trainingIterationData)
        {
            if(item is null)
                continue;
            data.Add([item.epoch, item.dataIndex, item.error, item.learningRate, item.elapsedSeconds]);
        }
        FilesCreatorHelper.CreateCsvFile(data, dirPath + "AllErrors.csv");
        data.Clear();

        if(trainEpochCorrectness.Length > 0)
        {
            data = [["Epoch", "Correctness", "AvgError", "MinError", "MaxError", "ElapsedSeconds"]];
            int dataLength = trainingIterationData.Length / trainEpochCorrectness.Length;
            for (int i = 0; i < trainEpochCorrectness.Length; i++)
            {
                var tmp = trainingIterationData.Where(x => x.epoch == i).ToArray();
                float avgError = tmp.Average(x => x.error);
                float minError = tmp.Min(x => x.error);
                float maxError = tmp.Max(x => x.error);
                float elapsedSeconds = tmp.Max(x => x.elapsedSeconds);
                data.Add([i, trainEpochCorrectness[i], avgError, minError, maxError, elapsedSeconds]);
            }
            FilesCreatorHelper.CreateCsvFile(data, dirPath + "EpochError.csv");
            data.Clear();
        }
        

        if(saveNN)
        {
            neuralNetwork.SaveToXmlFile(dirPath + "NeuralNetwork.xml");
        }

        var xml = FilesCreatorHelper.CreateXmlFile(dirPath + "TrainerConfig.xml");
        if(xml is not null)
        {
            xml.WriteStartElement("Root");
            xml.WriteStartElement("BaseConfig");
            xml.WriteElementString("InitialLearningRate", initialLearningRate.ToString());
            xml.WriteElementString("EpochAmount", epochAmount.ToString());
            xml.WriteElementString("BatchSize", batchSize.ToString());
            xml.WriteEndElement();

            if(isPatience)
            {
                xml.WriteStartElement("PatienceConfig");
                xml.WriteElementString("InitialIgnore", userInitialIgnore.ToString());
                xml.WriteElementString("Patience", userPatience.ToString());
                xml.WriteEndElement();
            }
            xml.WriteEndElement();
            FilesCreatorHelper.CloseXmlFile(xml);
        }
    }

}