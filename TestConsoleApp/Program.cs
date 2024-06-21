using System.Net;
using NeuralNetworkLibrary;
using ImagesProcessor;
using static NeuralNetworkLibrary.MatrixExtender;
using Accord.IO;

namespace TestConsoleApp;

internal class Program
{
    private const string MnistDataDirPath = "./../../../../../Datasets/Mnist/";

    private static void Main(string[] args)
    {
        //TODO perform tests (is new architecture better than old one? Is pooling layer correct? etc.)
        var tester = new NNTrainingTests();
        tester.RunTests();
        return;


    }

    private static void TestNN(NeuralNetwork nn)
    {
        nn.OnBatchTrainingIteration += (epoch, epochPercentFinish, error) =>
        {
            Console.WriteLine(
                            $"Epoch: {epoch + 1}\n" +
                            $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                            $"Batch error: {error.ToString("0.000")}\n" + 
                            $"Learning rate: {nn.LearningRate}\n");
        };

        const bool flatten = false;
        var trainData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        var testData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_test_data.csv");

        var trainer = new Trainer(nn, testData, 
            initialLearningRate: 0.01, 
            minLearningRate: 0.000001,
            epochAmount: 2, 
            batchSize: 50);

        trainer.SetPatience(
            initialIgnore: 0.3, 
            patience: 0.1,
            learningRateModifier: (lr) => lr * 0.9);

        var outputDir = trainer.SetLogSaving("./../../../../LearningLogs/", saveNN: true);

        Console.WriteLine("Training...");
        trainer.RunTraining();

        Console.WriteLine("FINAL Testing...");
        var correctness = nn.CalculateCorrectness(testData);

        var tmp = outputDir[..^1];
        Directory.Move(tmp, tmp + $"__{correctness.ToString("0.00")}");

        Console.WriteLine($"Correctness: {correctness.ToString("0.00")}%");

        //nn.SaveFeatureMaps(testData[0].input, "./../../../");
    }

    

    private static (Matrix input, Matrix expectedOutput)[] GetMnistDataMatrix(bool flatten, params string[] paths)
    {
        var results = new List<(Matrix, Matrix)>();
        int[] filter = []; //TODO remove after testing
        foreach (var path in paths)
        {
            var data = FilesCreatorHelper.ReadInputFromCSV(path, ',').Skip(1);
            foreach (var item in data)
            {
                Matrix tmpIn = new Matrix(28, 28);

                double[] input = item.Skip(1).Select(x => double.Parse(x) / 255.0).ToArray();
                double[] expected = new double[10];
                int numer = int.Parse(item[0]);
                expected[numer] = 1;

                if(filter.Contains(numer)) continue;

                Matrix tmpOut = new Matrix(expected);

                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        tmpIn[i, j] = input[i * 28 + j];
                    }
                }

                if(flatten)
                    results.Add((MatrixExtender.FlattenMatrix(tmpIn), tmpOut));
                else
                    results.Add((tmpIn, tmpOut));
            }
        }
        return results.ToArray();
    }
}