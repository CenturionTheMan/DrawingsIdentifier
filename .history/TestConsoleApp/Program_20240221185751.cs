using MyBaseLibrary;
using NeuralNetworkLibrary;


namespace TestConsoleApp;

class Program
{
    const string MnistDataDirPath = "D:\\GoogleDriveMirror\\Projects\\NeuralNetworkProject\\mnist_data\\";

    static void Main(string[] args)
    {
        Console.WriteLine("Loading data...");
        var trainData = GetMnistData(MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        var testData = GetMnistData(MnistDataDirPath + "mnist_test_data.csv");

        Console.WriteLine("Training...");
        var nn = new ConvolutionalNeuralNetwork([784, 16, 16, 10], [ActivationFunction.ReLU, ActivationFunction.ReLU, ActivationFunction.Softmax]);

        nn.Train(trainData, 0.01, 30, 200, 0.01);

        Console.WriteLine("Testing...");
        int guessed = 0;
        foreach (var item in testData)
        {
            var prediction = nn.Predict(item.inputs);
            var max = prediction.Max();
            int indexOfMaxPrediction = prediction.ToList().IndexOf(max);

            var expectedMax = item.outputs.Max();
            int indexOfMaxExpected = item.outputs.ToList().IndexOf(expectedMax);

            if(indexOfMaxPrediction == indexOfMaxExpected)
            {
                guessed++;
            }
        }

        Console.WriteLine($"Correctness: {(guessed*100.0/(double)testData.Length).ToString("0.00")}%");
    }

    private static (double[] inputs, double[] outputs)[] GetMnistData(params string[] paths)
    {
        var result = new List<(double[] inputs, double[] outputs)>();

        foreach (var path in paths)
        {
            var data = FilesCreatorHelper.ReadInputFromCSV(path, ',').Skip(1);

            foreach (var item in data)
            {
                double[] inputs = new double[784];
                double[] expected = new double[10];

                int numer = int.Parse(item[0]);
                expected[numer] = 1;

                inputs = item.Skip(1).Select(x => double.Parse(x)/255.0).ToArray();

                result.Add((inputs, expected));
            }
        }

        return result.ToArray();
    }
}