using System.Net;
using MyBaseLibrary;
using NeuralNetworkLibrary;
using ImagesProcessor;
using static NeuralNetworkLibrary.MatrixExtender;

namespace TestConsoleApp;

internal class Program
{
    private const string MnistDataDirPath = "D:\\GoogleDriveMirror\\Projects\\NeuralNetworkProject\\mnist_data\\";

    private static void Main(string[] args)
    {
        //TestCNN(
        //    new ConvolutionalNeuralNetwork(
        //    [
        //        new ConvolutionLayer((1, 28, 28), 3, 4, ActivationFunction.Sigmoid),
        //        new ConvolutionLayer((4, 26, 26), 3, 2, ActivationFunction.Sigmoid),
        //    ],
        //    [
        //        new FullyConnectedLayer(16, ActivationFunction.ReLU, 24*24*2),
        //        new FullyConnectedLayer(16, ActivationFunction.ReLU, 16),
        //        new FullyConnectedLayer(10, ActivationFunction.Softmax, 16)
        //    ]), 0.1, 20, 50
        //);

        TestNN(new NeuralNetwork(28*28, [
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax)

        ]), 1, 1, 50);

        // TestNN();
    }

    private static void TestNN(NeuralNetwork cnn, double learningRate, int epochAmount, int batchSize)
    {
        Console.WriteLine("Loading data...");

        const bool flatten = true;
        var trainData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        var testData = GetMnistDataMatrix(flatten, MnistDataDirPath + "mnist_test_data.csv");

        Console.WriteLine("Training...");

        cnn.OnBatchLearningIteration += (epoch, epochPercentFinish, error) =>
        {
            Console.WriteLine(
                            $"Epoch: {epoch + 1}\n" +
                            $"Epoch percent finish: {epochPercentFinish.ToString("0.00")}%\n" +
                            $"Batch error: {error.ToString("0.000")}\n");
        };

        cnn.Train(trainData, learningRate, epochAmount, batchSize);


        Console.WriteLine("Testing...");
        var correctness = cnn.CalculateCorrectness(testData);
        Console.WriteLine($"Correctness: {correctness.ToString("0.00")}%");
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