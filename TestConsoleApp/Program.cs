using System.Net;
using MyBaseLibrary;
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

        // var nn = new NeuralNetwork(1, 28, 28, [
        //     LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 4, stride: 1, activationFunction: ActivationFunction.ReLU),
        //     LayerTemplate.CreatePoolingLayer(poolSize: 2, stride: 2),
        //     LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 8, stride: 1, activationFunction: ActivationFunction.ReLU),
        //     LayerTemplate.CreatePoolingLayer(poolSize: 2, stride: 2),
        //     LayerTemplate.CreateFullyConnectedLayer(layerSize: 100, activationFunction: ActivationFunction.ReLU),
        //     LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        // ]);
        // //82.24% correctness
        // TestNN(nn, new LearningScheduler(0.001, 5, 200, 1));

        //88% correctness
        var nn = new NeuralNetwork(1, 28, 28, [
            LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 8, stride: 1, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        ]);
        TestNN(nn, new LearningScheduler(0.01, 1, 50, 1));
    }

    private static void TestNN(NeuralNetwork nn, LearningScheduler learningScheduler)
    {
        const bool flatten = false;
        var trainData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        var testData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_test_data.csv");

        Console.WriteLine("Training...");

        CancellationTokenSource cts = new CancellationTokenSource();

        nn.OnBatchLearningIteration += (epoch, epochPercentFinish, error) =>
        {
            Console.WriteLine(
                            $"Epoch: {epoch + 1}\n" +
                            $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                            $"Batch error: {error.ToString("0.000")}\n" + 
                            $"Learning rate: {nn.LearningRate}\n");
        };

        nn.Train(trainData, learningScheduler);
       

        Console.WriteLine("FINAL Testing...");
        var correctness = nn.CalculateCorrectness(testData);
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