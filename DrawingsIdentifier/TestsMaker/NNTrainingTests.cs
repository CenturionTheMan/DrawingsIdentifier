using NeuralNetworkLibrary;
using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.NeuralNetwork;
using NeuralNetworkLibrary.QuickDrawHandler;
using NeuralNetworkLibrary.Utils;
using SixLabors.ImageSharp;

namespace TestConsoleApp;

internal class NNTrainingTests
{
    private const string MnistDataDirPath = "./../../../../../../Datasets/Mnist/";
    private const string QuickDrawDirPath = "./../../../../../../Datasets/QuickDraw/";

    private int counter = 0;

    private static Random random = new Random();

    (Matrix[] inputChannels, Matrix output)[] trainData;
    (Matrix[] inputChannels, Matrix output)[] testData;
    (Matrix[] inputChannels, Matrix output)[] trainDataFlatten;
    (Matrix[] inputChannels, Matrix output)[] testDataFlatten;

    (Matrix[] inputChannels, Matrix output)[] trainDataMnist;
    (Matrix[] inputChannels, Matrix output)[] testDataMnist;
    (Matrix[] inputChannels, Matrix output)[] trainDataFlattenMnist;
    (Matrix[] inputChannels, Matrix output)[] testDataFlattenMnist;

    public void RunTests()
    {
        //TODO test: https://github.com/amelie-vogel/image-classification-quickdraw
        Console.WriteLine("Loading data...");
        QuickDrawSet qds = QuickDrawDataReader.LoadQuickDrawSamplesFromDirectory(QuickDrawDirPath, amountToLoadFromEachFile: 10000, randomlyShift: false);
        (trainData, testData) = qds.SplitIntoTrainTest();

        //RESHAPE INTO SINGLE DIMENSION
        trainDataFlatten = trainData.Select(x => (new Matrix[] { MatrixExtender.FlattenMatrix(x.inputChannels) }, x.output)).ToArray();
        testDataFlatten = testData.Select(x => (new Matrix[] { MatrixExtender.FlattenMatrix(x.inputChannels) }, x.output)).ToArray();


        trainDataMnist = GetMnistDataMatrix(false, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        testDataMnist = GetMnistDataMatrix(false, MnistDataDirPath + "mnist_test_data.csv");
        trainDataFlattenMnist = GetMnistDataMatrix(true, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        testDataFlattenMnist = GetMnistDataMatrix(true, MnistDataDirPath + "mnist_test_data.csv");


        Console.WriteLine("Running tests");
        //for (int i = 0; i < repAmount; i++)
        //{
        //    SingleConvConstTrainerMnist(new NeuralNetwork(1, 28, 28, new LayerTemplate[]
        //    {
        //        LayerTemplate.CreateConvolutionLayer(5, 8, activationFunction: ActivationFunction.ReLU),
        //        LayerTemplate.CreateMaxPoolingLayer(2,2),
        //        LayerTemplate.CreateConvolutionLayer(3, 16, activationFunction: ActivationFunction.ReLU),
        //        LayerTemplate.CreateMaxPoolingLayer(2,2),
        //        LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //        LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        //    }), groupNum: 1, 20.0f);

        //    SingleConvConstTrainerMnist(new NeuralNetwork(1, 28, 28, new LayerTemplate[]
        //    {
        //    LayerTemplate.CreateConvolutionLayer(5, 16, activationFunction: ActivationFunction.ReLU),
        //    LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        //    }), groupNum: 2, 20.0f);

        //    SingleMLPConstTrainerMnist(new NeuralNetwork(784, new LayerTemplate[]
        //    {
        //    LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //    LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //    LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        //    }), groupNum: 3, 20.0f);
        //}
        //return;

        const int repAmount = 7;

        for (int i = 0; i < repAmount; i++)
        {

            SingleConvConstTrainer(new NeuralNetwork(1, 28, 28, new LayerTemplate[]
            {
            LayerTemplate.CreateConvolutionLayer(5, 8, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateMaxPoolingLayer(2,2),
            LayerTemplate.CreateConvolutionLayer(3, 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateMaxPoolingLayer(2,2),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 128, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
            }), groupNum: 1, 20.0f);

            SingleConvConstTrainer(new NeuralNetwork(1, 28, 28, new LayerTemplate[]
            {
            LayerTemplate.CreateConvolutionLayer(5, 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateConvolutionLayer(3, 32, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
            }), groupNum: 2, 20.0f);

            SingleMLPConstTrainer(new NeuralNetwork(784, new LayerTemplate[]
            {
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 128, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 128, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
            }), groupNum: 3, 20.0f);
        }
    }

    private void SingleMLPConstTrainer(NeuralNetwork nn, int groupNum, float minExpectedCorrectness = 20.0f)
    {
        Single(
            testData: testDataFlatten,

            trainer: new Trainer(
                nn,
                data: trainDataFlatten,
                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 5, batchSize: 100)
                .SetAutoReinitialize(minExpectedCorrectness, 3),
            groupNum
        );
    }

    private void SingleConvConstTrainer(NeuralNetwork nn, int groupNum, float minExpectedCorrectness = 20.0f)
    {
        Single(
            testData: testData,

            trainer: new Trainer(
                nn,
                data: trainData,
                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 5, batchSize: 100)
                .SetAutoReinitialize(minExpectedCorrectness, 3
            ),
            groupNum
        );
    }

    private void SingleMLPConstTrainerMnist(NeuralNetwork nn, int groupNum, float minExpectedCorrectness = 20.0f)
    {
        Single(
            testData: testDataFlattenMnist,

            trainer: new Trainer(
                nn,
                data: trainDataFlattenMnist,
                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 5, batchSize: 100)
                //.SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.002f)
                .SetAutoReinitialize(minExpectedCorrectness, 3),
            groupNum
        );
    }

    private void SingleConvConstTrainerMnist(NeuralNetwork nn, int groupNum, float minExpectedCorrectness = 20.0f)
    {
        Single(
            testData: testDataMnist,

            trainer: new Trainer(
                nn,
                data: trainDataMnist,
                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 5, batchSize: 100)
                //.SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.002f)
                .SetAutoReinitialize(minExpectedCorrectness, 3
            ),
            groupNum
        );
    }

    private void Single((Matrix[] inputChannels, Matrix output)[] testData, Trainer trainer, int groupNum)
    {
        Console.WriteLine($"Test {++counter}");

        trainer.SetLogSaving($"./../../../../../../LearningLogs/GROUP_TEST_{groupNum}/", saveNN: true, testData: testData, out string outputDir);

        var samples = testData.OrderBy(x => random.Next()).Take(1000).ToArray();
        float sampleCorrectness = trainer.NeuralNetwork.CalculateCorrectness(samples);

        trainer.NeuralNetwork.OnBatchTrainingIteration += (epoch, epochPercentFinish, error) =>
        {
            Console.WriteLine(
                            $"Epoch: {epoch}\n" +
                            $"Epoch percent finish: {epochPercentFinish:0.00}%\n" +
                            $"Batch error: {error:0.000}\n" +
                            $"Learning rate: {trainer.NeuralNetwork.LearningRate}\n" +
                            $"Last correctness: {sampleCorrectness:0.00}%\n"
                            );
        };

        trainer.NeuralNetwork.OnEpochTrainingIteration += (epoch, correctnes_) =>
        {
            sampleCorrectness = trainer.NeuralNetwork.CalculateCorrectness(samples);
        };

        Console.WriteLine("Training...");
        trainer.RunTraining();

        Console.WriteLine("FINAL Testing...");
        var correctness = trainer.NeuralNetwork.CalculateCorrectness(testData);

        Console.WriteLine($"Moving logs to new directory...");
        var tmp = outputDir[..^1];
        Directory.Move(tmp, tmp + $"__{correctness:0.00}");
        outputDir = tmp + $"__{correctness:0.00}/";

        Console.WriteLine($"Correctness: {correctness:0.00}%");

        string mapsDir = outputDir + "FeatureMaps";
        if (!Directory.Exists(mapsDir))
            Directory.CreateDirectory(mapsDir);
        trainer.NeuralNetwork.SaveFeatureMaps(testData[0].inputChannels, mapsDir + "/");

        Console.WriteLine();
    }


    private static (Matrix[] inputChannels, Matrix expectedOutput)[] GetMnistDataMatrix(bool flatten, params string[] paths)
    {
        var results = new List<(Matrix[], Matrix)>();
        int[] filter = [];
        foreach (var path in paths)
        {
            var data = FilesCreatorHelper.ReadInputFromCSV(path, ',').Skip(1);
            foreach (var item in data)
            {
                Matrix tmpIn = new Matrix(28, 28);

                float[] input = item.Skip(1).Select(x => (float)(float.Parse(x) / 255.0)).ToArray();
                float[] expected = new float[10];
                int numer = int.Parse(item[0]);
                expected[numer] = 1;

                if (filter.Contains(numer)) continue;

                Matrix tmpOut = new Matrix(expected);

                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        tmpIn[i, j] = input[i * 28 + j];
                    }
                }

                if (flatten)
                    results.Add(([MatrixExtender.FlattenMatrix(tmpIn)], tmpOut));
                else
                    results.Add(([tmpIn], tmpOut));
            }
        }
        return results.ToArray();
    }
}