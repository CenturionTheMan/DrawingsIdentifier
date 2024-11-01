﻿using System.Net;
using NeuralNetworkLibrary;
using static NeuralNetworkLibrary.MatrixExtender;
using System.Diagnostics;

namespace TestConsoleApp;

internal class Program
{
    private const string MnistDataDirPath = "./../../../../../Datasets/Mnist/";

    private static void Main(string[] args)
    {
        //TODO perform tests (is new architecture better than old one? Is pooling layer correct? etc.)
        //var tester = new NNTrainingTests();
        //tester.RunTests();
        //throw new Exception("END");

        TestNN(new NeuralNetwork(784, new LayerTemplate[]
        {
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        }));

    }

    private static void TestNN(NeuralNetwork nn)
    {
        Stopwatch stopwatch = new Stopwatch();

        nn.OnBatchTrainingIteration += (epoch, epochPercentFinish, error) =>
        {
            Console.WriteLine(
                            $"Epoch: {epoch + 1}\n" +
                            $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                            $"Batch error: {error.ToString("0.000")}\n" +
                            $"Learning rate: {nn.LearningRate}\n");
        };

        Console.WriteLine("Loading data...");
        const bool flatten = true;
        var trainData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        var testData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_test_data.csv");

        Console.WriteLine("Training...");

        const float learningRate = 0.01f;
        const int epochAmount = 3;
        const int batchSize = 50;

        stopwatch.Start();
        nn.Train(trainData, learningRate, epochAmount, batchSize);
        stopwatch.Stop();

        Console.WriteLine($"\n Elapsed time: {stopwatch.Elapsed.TotalSeconds.ToString("0.00")}s\n Avg time per sample: {(stopwatch.Elapsed.TotalSeconds / (trainData.Length * epochAmount)).ToString("0.0000")}s\n");

        Console.WriteLine("FINAL Testing...");
        var correctness = nn.CalculateCorrectness(testData);
        Console.WriteLine($"Correctness: {correctness.ToString("0.00")}%");

        nn.SaveToXmlFile("./../../../nn.xml", null);
        var nnLod = NeuralNetwork.LoadFromXmlFile("./../../../nn.xml");
        Console.WriteLine($"Loaded NN correctness: {nnLod!.CalculateCorrectness(testData).ToString("0.00")}%");

        //nn.SaveFeatureMaps(testData[0].input, "./../../../");
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