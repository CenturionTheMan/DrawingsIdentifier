using ImagesProcessor;

namespace NeuralNetworkLibrary;

class NNTrainingTests
{
    private const string MnistDataDirPath = "./../../../../../Datasets/Mnist/";
    private const string QuickDrawDirPath = "./../../../../../Datasets/QuickDraw/";

    private int counter = 0;

    public void RunTests()
    {
        //TODO test: https://github.com/amelie-vogel/image-classification-quickdraw
        
        // (Matrix input, Matrix output)[] trainData = GetMnistDataMatrix(false, MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv"); 
        // (Matrix input, Matrix output)[] testData = GetMnistDataMatrix(false, MnistDataDirPath + "mnist_test_data.csv");

        ((Matrix input, Matrix output)[] trainData, (Matrix input, Matrix output)[] testData) = GetQuickDrawDataMatrix(QuickDrawDirPath, 5000);

        //! TEST 1
        var nn = new NeuralNetwork(1, 28, 28,
        [
            LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 8, stride: 1, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreatePoolingLayer(poolSize: 2, stride: 2),
            LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 16, stride: 1, activationFunction: ActivationFunction.Sigmoid),
            LayerTemplate.CreatePoolingLayer(poolSize: 2, stride: 2),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 100, activationFunction: ActivationFunction.ReLU),
            LayerTemplate.CreateFullyConnectedLayer(layerSize: 10, activationFunction: ActivationFunction.Softmax),
        ]);
        Trainer trainer = new Trainer(nn, trainData, 
            initialLearningRate: 0.01, 
            minLearningRate: 0.001, 
            epochAmount: 20, 
            batchSize: 50);
        trainer.SetPatience(initialIgnore: 0.2, patience: 0.1, learningRateModifier: (lr) => lr * 0.9);
        Single(testData, nn, trainer);
    }

    private void Single((Matrix input, Matrix output)[] testData, NeuralNetwork nn, Trainer trainer)
    {
        Console.WriteLine($"Test {++counter}");

        var outputDir = trainer.SetLogSaving("./../../../../LearningLogs/", saveNN: true);

        nn.OnBatchTrainingIteration += (epoch, epochPercentFinish, error) =>
        {
            Console.WriteLine(
                            $"Epoch: {epoch + 1}\n" +
                            $"Epoch percent finish: {epochPercentFinish:0.00}%\n" +
                            $"Batch error: {error:0.000}\n" + 
                            $"Learning rate: {nn.LearningRate}\n");
        };

        Console.WriteLine("Training...");
        trainer.RunTraining();

        Console.WriteLine("FINAL Testing...");
        var correctness = nn.CalculateCorrectness(testData);

        var tmp = outputDir[..^1];
        Directory.Move(tmp, tmp + $"__{correctness:0.00}");
        outputDir = tmp + $"__{correctness:0.00}/";

        Console.WriteLine($"Correctness: {correctness:0.00}%");

        string mapsDir = outputDir + "FeatureMaps";
        if(!Directory.Exists(mapsDir))
            Directory.CreateDirectory(mapsDir);
        nn.SaveFeatureMaps(testData[0].input, mapsDir + "/");

        Console.WriteLine();
    }



    private static ((Matrix input, Matrix expectedOutput)[] train, (Matrix input, Matrix expectedOutput)[] test) GetQuickDrawDataMatrix(string pathDir, int samplesPerFile)
    {
        QuickDrawSet qds = ImagesProcessor.DataReader.LoadQuickDrawSamplesFromDirectory(pathDir, amountToLoadFromEachFile: samplesPerFile);
        var rawSplit = qds.SplitIntoTrainTest();

        (Matrix input, Matrix expectedOutput)[] train = new (Matrix input, Matrix expectedOutput)[rawSplit.trainData.Length]; 
        (Matrix input, Matrix expectedOutput)[] test = new (Matrix input, Matrix expectedOutput)[rawSplit.testData.Length];
        for (int i = 0; i < rawSplit.trainData.Length; i++)
        {
            var sample = rawSplit.trainData[i];
            var tmp = new Matrix(sample.inputs);
            var unflatten = MatrixExtender.UnflattenMatrix(tmp, 28, 28);
            if(unflatten.Length > 1)
                throw new Exception("Unflatten matrix has more than one element");
            tmp = unflatten[0];
            train[i] = (tmp, new Matrix(sample.outputs));   
        }

        for (int i = 0; i < rawSplit.testData.Length; i++)
        {
            var sample = rawSplit.testData[i];
            var tmp = new Matrix(sample.inputs);
            var unflatten = MatrixExtender.UnflattenMatrix(tmp, 28, 28);
            if(unflatten.Length > 1)
                throw new Exception("Unflatten matrix has more than one element");
            tmp = unflatten[0];
            test[i] = (tmp, new Matrix(sample.outputs));   
        }

        return (train, test);
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