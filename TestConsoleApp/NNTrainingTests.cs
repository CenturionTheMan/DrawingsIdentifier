using NeuralNetworkLibrary;
using NeuralNetworkLibrary.QuickDrawHandler;
using SixLabors.ImageSharp;

namespace TestConsoleApp;

internal class NNTrainingTests
{
    private const string MnistDataDirPath = "./../../../../../Datasets/Mnist/";
    private const string QuickDrawDirPath = "./../../../../../Datasets/QuickDraw/";

    private int counter = 0;

    private static Random random = new Random();

    public void RunTests()
    {
        //TODO test: https://github.com/amelie-vogel/image-classification-quickdraw

        QuickDrawSet qds = QuickDrawDataReader.LoadQuickDrawSamplesFromDirectory(QuickDrawDirPath, amountToLoadFromEachFile: 10000, randomlyShift: true);
        ((Matrix[] inputChannels, Matrix output)[] trainData, (Matrix[] inputChannels, Matrix output)[] testData) = qds.SplitIntoTrainTest();

        //RESHAPE INTO SINGLE DIMENSION
        var trainDataFlatten = trainData.Select(x => (new Matrix[] { MatrixExtender.FlattenMatrix(x.inputChannels) }, x.output)).ToArray();
        var testDataFlatten = testData.Select(x => (new Matrix[] { MatrixExtender.FlattenMatrix(x.inputChannels) }, x.output)).ToArray();

        //! TEST 1 >>
        // Single(
        //     testData: testData,

        //     trainer: new Trainer(
        //         new NeuralNetwork(1, 28, 28,
        //         [
        //             LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 10, stride: 1, activationFunction: ActivationFunction.ReLU),
        //             LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //             LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 20, stride: 1, activationFunction: ActivationFunction.Sigmoid),
        //             LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //             LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //             LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //             LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        //         ]),

        //         data: trainData,

        //         initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 30, batchSize: 50)
        //         .SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.001f
        //     )
        // );

        //! TEST 2 >>
        Single(
            testData: testData,

            trainer: new Trainer(
                new NeuralNetwork(1, 28, 28,
                [
                    LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 16, stride: 1, activationFunction: ActivationFunction.ReLU),
                    LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                    LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
                    LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
                    LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
                ]),

                data: trainData,

                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 30, batchSize: 50)
                .SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.001f
            )
        );
    }

    private void Single((Matrix[] inputChannels, Matrix output)[] testData, Trainer trainer)
    {
        Console.WriteLine($"Test {++counter}");

        trainer.SetLogSaving("./../../../../LearningLogs/GROUP_TEST_1/", saveNN: true, testData: testData, out string outputDir);

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

        trainer.NeuralNetwork.OnEpochTrainingIteration += (epoch, error) =>
        {
            sampleCorrectness = trainer.NeuralNetwork.CalculateCorrectness(samples);
        };

        Console.WriteLine("Training...");
        (Task task, CancellationTokenSource cts) = trainer.RunTrainingOnTask();
        Task.Factory.StartNew(async () =>
        {
            while (task.Status == TaskStatus.Running && cts.IsCancellationRequested == false)
            {
                if (Console.In.Peek() != -1)
                {
                    if (Console.ReadLine()!.ToLower().Contains("q"))
                    {
                        cts.Cancel();
                        break;
                    }
                }
                await Task.Delay(1000);
            }
        });
        task.Wait();

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
}