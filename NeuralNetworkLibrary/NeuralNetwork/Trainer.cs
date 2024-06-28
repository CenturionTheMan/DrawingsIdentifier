using System.Diagnostics;

namespace NeuralNetworkLibrary;

public class Trainer
{
    private NeuralNetwork neuralNetwork;
    private (Matrix[] inputChannels, Matrix output)[] data;
    private float initialLearningRate;
    private float minLearningRate;
    private int epochAmount;
    private int batchSize;

    private bool isPatience = false;
    private float userInitialIgnore = 0.0f;
    private float userPatience = 0;
    private Queue<(float max, float avg)>? lastErrorsRange;
    private float initialIgnore = 10.0f; //percent
    private int patienceAmount = 50;
    private bool hasCrossedIgnoreThreshold = false;
    private Func<float, float> learningRateModifier = (lr) => lr * 0.9f;

    private bool saveToLog = false;
    private bool saveNN = false;
    private string trainingLogDir = "";
    
    private record TrainingIterationData(int epoch, int dataIndex, float error, float learningRate, float elapsedSeconds);


    public Trainer(NeuralNetwork neuralNetwork, (Matrix[] inputChannels, Matrix output)[] data, float initialLearningRate, float minLearningRate, int epochAmount, int batchSize)
    {
        this.neuralNetwork = neuralNetwork;
        this.data = data;
        this.initialLearningRate = initialLearningRate;
        this.minLearningRate = minLearningRate;
        this.epochAmount = epochAmount;
        this.batchSize = batchSize;
    }

    public void SetPatience(float initialIgnore, float patience, Func<float, float>? learningRateModifier = null)
    {
        initialIgnore = Math.Clamp(initialIgnore, 0, 1);
        patience = Math.Clamp(patience, 0, 1);
        
        this.userInitialIgnore = initialIgnore;
        this.userPatience = patience;

        this.initialIgnore = initialIgnore * 100;
        this.patienceAmount = (int)(data.Length * patience / batchSize);
        this.patienceAmount = Math.Max(1, patienceAmount);
        this.lastErrorsRange = new(patienceAmount);

        if(learningRateModifier is not null)
            this.learningRateModifier = learningRateModifier;

        isPatience = true;
    }

    public string SetLogSaving(string outputDirPath, bool saveNN)
    {
        if(!Directory.Exists(outputDirPath))
            Directory.CreateDirectory(outputDirPath);

        this.trainingLogDir = outputDirPath + DateTime.Now.ToString("yyyy.MM.dd__HH-mm-ss");

        if(!Directory.Exists(trainingLogDir))
            Directory.CreateDirectory(trainingLogDir);
        this.trainingLogDir += "/";

        this.saveNN = saveNN;
        this.saveToLog = true;
        return this.trainingLogDir;
    }

    public void RunTraining()
    {
        var stopwatch = new Stopwatch();

        var cts = new CancellationTokenSource();

        Stack<TrainingIterationData> trainingIterationData = new();
        List<float> trainCorrectness = new List<float>(epochAmount);

        Queue<float> lastAvgBatchErrors = new Queue<float>(patienceAmount);

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
            lastAvgBatchErrors.Enqueue(batchAvgError);
            if(lastAvgBatchErrors.Count > patienceAmount)
            {
                lastAvgBatchErrors.Dequeue();
            }

            if(isPatience)
                HandlePatience(neuralNetwork, lastAvgBatchErrors, epochPercentFinish);
        };

        neuralNetwork.OnEpochTrainingIteration += (epoch, correctness) =>
        {
            trainCorrectness.Add(correctness);

            if(neuralNetwork.LearningRate <= minLearningRate)
            {
                cts?.Cancel();
            }
        };

        neuralNetwork.OnTrainingFinished += () =>
        {
 
        };

        stopwatch.Start();
        neuralNetwork.Train(data, initialLearningRate, epochAmount, batchSize, cts.Token);
        stopwatch.Stop();
        
        if(saveToLog)
            SaveTrainingData(trainingLogDir, trainingIterationData.ToArray(), trainCorrectness.ToArray());
    }

    private void HandlePatience(NeuralNetwork nn, Queue<float> lastAvgBatchErrors, float epochPercentFinish)
    {
        if(hasCrossedIgnoreThreshold == false || nn.LearningRate <= minLearningRate)
        {
            if(epochPercentFinish >= initialIgnore)
                hasCrossedIgnoreThreshold = true;
            
            return;
        }

        var max = lastAvgBatchErrors.Max();
        var avg = lastAvgBatchErrors.Average();
        lastErrorsRange!.Enqueue((max, avg));
        if(lastErrorsRange.Count() < patienceAmount)return;

        var avgMax = lastErrorsRange.Average(x => x.max);
        var avgAvg = lastErrorsRange.Average(x => x.avg);

        if(max >= avgMax && avg >= avgAvg)
        {
            nn.LearningRate = Math.Max(minLearningRate, learningRateModifier(nn.LearningRate));
        }
        lastErrorsRange?.Clear();
    }

    private void SaveTrainingData(string dirPath, TrainingIterationData[] trainingIterationData, float[] trainEpochCorrectness)
    {
        trainingIterationData = trainingIterationData.Reverse().ToArray();

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
        }
        

        neuralNetwork.SaveToXmlFile(dirPath + "NeuralNetwork.xml");

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