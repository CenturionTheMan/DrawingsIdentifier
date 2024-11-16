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


    public void RunTests()
    {
        //TODO test: https://github.com/amelie-vogel/image-classification-quickdraw
        Console.WriteLine("Loading data...");
        QuickDrawSet qds = QuickDrawDataReader.LoadQuickDrawSamplesFromDirectory(QuickDrawDirPath, amountToLoadFromEachFile: 10000, randomlyShift: true);
        (trainData, testData) = qds.SplitIntoTrainTest();

        //RESHAPE INTO SINGLE DIMENSION
        trainDataFlatten = trainData.Select(x => (new Matrix[] { MatrixExtender.FlattenMatrix(x.inputChannels) }, x.output)).ToArray();
        testDataFlatten = testData.Select(x => (new Matrix[] { MatrixExtender.FlattenMatrix(x.inputChannels) }, x.output)).ToArray();
        Console.WriteLine("Running tests");


        //Single(
        //    testData: testData,

        //    trainer: new Trainer(
        //        new NeuralNetwork(1, 28, 28,
        //        [
        //            LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 2, activationFunction: ActivationFunction.ReLU),
        //            LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //            LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 4, activationFunction: ActivationFunction.Sigmoid),
        //            LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //            LayerTemplate.CreateFullyConnectedLayer(layerSize: 32, activationFunction: ActivationFunction.ReLU),
        //            LayerTemplate.CreateFullyConnectedLayer(layerSize: 32, activationFunction: ActivationFunction.ReLU),
        //            LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
        //        ]),
        //        data: trainData,
        //        initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 1, batchSize: 50)
        //        .SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.001f
        //    )
        //);


        //for (int i = 8; i <= 16; i += 8)
        //{
        //    for (int j = 8; j <= 32; j += 8)
        //    {
        //        SingleConvConstTrainer(
        //            new NeuralNetwork(1, 28, 28,
        //            [
        //                LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: i, activationFunction: ActivationFunction.ReLU),
        //                LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //                LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: j, activationFunction: ActivationFunction.Sigmoid),
        //                LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //                LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
        //                LayerTemplate.CreateFullyConnectedLayer(layerSize: 16, activationFunction: ActivationFunction.ReLU),
        //                LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
        //            ]),
        //            groupNum: 1)
        //        ;
        //    }
        //}

        //for (int j = 32; j <= 32; j += 4)
        //{
        //    SingleConvConstTrainer(
        //        new NeuralNetwork(1, 28, 28,
        //        [
        //            LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: j, activationFunction: ActivationFunction.ReLU),
        //            LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
        //            LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //            LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
        //            LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
        //        ]),
        //        groupNum: 2
        //    );
        //}

        //for (int i = 16; i <= 64; i += 16)
        //{
        //    for (int j = 16; j <= 64; j += 16)
        //    {
        //        SingleMLPConstTrainer(
        //            new NeuralNetwork(784,
        //            [
        //                LayerTemplate.CreateFullyConnectedLayer(layerSize: i, activationFunction: ActivationFunction.ReLU),
        //                LayerTemplate.CreateFullyConnectedLayer(layerSize: j, activationFunction: ActivationFunction.ReLU),
        //                LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
        //            ]),
        //            groupNum: 3
        //        );
        //    }
        //}
        const int rep = 2;


        for (int r = 0; r < rep; r++)
        {
            for (int i = 5; i <= 15; i += 5)
            {
                SingleConvConstTrainer(
                    new NeuralNetwork(1, 28, 28,
                    [
                        LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: i, activationFunction: ActivationFunction.ReLU),
                        LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                        LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 20, activationFunction: ActivationFunction.Sigmoid),
                        LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                        LayerTemplate.CreateFullyConnectedLayer(layerSize: 60, activationFunction: ActivationFunction.ReLU),
                        LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
                    ]),
                    groupNum: 1,
                    minExpectedCorrectness: 20.0f
                );
            }
        }

        for (int r = 0; r < rep; r++)
        {
            for (int i = 15; i <= 25; i += 5)
            {
                SingleConvConstTrainer(
                    new NeuralNetwork(1, 28, 28,
                    [
                        LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 10, activationFunction: ActivationFunction.ReLU),
                        LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                        LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: i, activationFunction: ActivationFunction.Sigmoid),
                        LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                        LayerTemplate.CreateFullyConnectedLayer(layerSize: 60, activationFunction: ActivationFunction.ReLU),
                        LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
                    ]),
                    groupNum: 2,
                    minExpectedCorrectness: 20.0f
                );
            }
        }

        for (int r = 0; r < rep; r++)
        {
            for (int i = 30; i <= 90; i += 30)
            {
                SingleConvConstTrainer(
                    new NeuralNetwork(1, 28, 28,
                    [
                        LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 10, activationFunction: ActivationFunction.ReLU),
                        LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                        LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 20, activationFunction: ActivationFunction.Sigmoid),
                        LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
                        LayerTemplate.CreateFullyConnectedLayer(layerSize: i, activationFunction: ActivationFunction.ReLU),
                        LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax),
                    ]),
                    groupNum: 3,
                    minExpectedCorrectness: 20.0f
                );
            }
        }
    }

    private void SingleMLPConstTrainer(NeuralNetwork nn, int groupNum, float minExpectedCorrectness = 20.0f)
    {
        Single(
            testData: testDataFlatten,

            trainer: new Trainer(
                nn,
                data: trainDataFlatten,
                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 10, batchSize: 100)
                .SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.002f)
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
                initialLearningRate: 0.01f, minLearningRate: 0.0001f, epochAmount: 10, batchSize: 100)
                .SetPatience(initialIgnore: 0.9f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.002f)
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
}