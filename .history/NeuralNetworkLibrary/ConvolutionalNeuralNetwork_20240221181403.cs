namespace NeuralNetworkLibrary;

public class ConvolutionalNeuralNetwork
{
    private static Random random = new Random();

    private int[] layersSizes;
    private Matrix[] biasesForLayers;
    private Matrix[] weightsForLayers;
    private Matrix[] layersBeforeActivation;
    private double learningRate;
    private ActivationFunction[] activationFunctions;
    private int layersAmount;

    public ConvolutionalNeuralNetwork(int[] layersSizes, ActivationFunction[] activationFunctions)
    {
        if(layersSizes.Length != activationFunctions.Length + 1)
        {
            throw new ArgumentException("Activation functions amount must be equal to layers amount - 1");
        }

        this.layersSizes = layersSizes;
        this.layersAmount = layersSizes.Length;

        this.activationFunctions = activationFunctions;

        this.layersBeforeActivation = new Matrix[layersAmount];
        this.biasesForLayers = new Matrix[layersAmount - 1];
        this.weightsForLayers = new Matrix[layersAmount - 1];


        for (int i = 0; i < layersAmount; i++)
        {
            this.layersBeforeActivation[i] = new Matrix(layersSizes[i], 1);
        }

        for (int i = 0; i < layersAmount - 1; i++)
        {
            int fromLayer = layersSizes[i];
            int toLayer = layersSizes[i + 1];
            this.weightsForLayers[i] = new Matrix(toLayer, fromLayer, -0.25, 0.25);
            this.biasesForLayers[i] = new Matrix(toLayer, 1, -0.1, 0.1);
        }
    }

    public void Train((double[] inputs, double[] outputs)[] data, double learningRate, int epochAmount, int batchSize, double expectedMaxError = 0.001)
    {
        if (data[0].inputs.Length != layersSizes[0])
        {
            throw new ArgumentException("Inputs length must be equal to the first layer size");
        }

        if (data[0].outputs.Length != layersSizes[layersAmount - 1])
        {
            throw new ArgumentException("Outputs length must be equal to the last layer size");
        }

        this.learningRate = learningRate;

        for (int epoch = 0; epoch < epochAmount; epoch++)
        {
            data = data.OrderBy(x => random.Next()).ToArray();
            int batchBeginIndex = 0;

            while (batchBeginIndex < data.Length)
            {
                var batchSamples = batchBeginIndex + batchSize < data.Length ? data.Skip(batchBeginIndex).Take(batchSize) : data[batchBeginIndex..];

                Matrix[] inputSamples = batchSamples.Select(x => new Matrix(x.inputs)).ToArray()!;
                Matrix[] expectsOutputsSamples = batchSamples.Select(x => new Matrix(x.outputs)).ToArray()!;

                double batchError = PerformLearningIteration(inputSamples, expectsOutputsSamples);

                Console.WriteLine($"Epoch: {epoch+1}\n" +
                                  $"Epoch percent finish: {(100*batchBeginIndex/(double)data.Length).ToString("0.00")}%\n" +
                                  $"Batch error: {batchError.ToString("0.000")}\n");

                if(batchError < expectedMaxError)
                {
                    return;
                }

                batchBeginIndex += batchSize;
            }
        }
    }

    public double[] Predict(double[] inputs)
    {
        if(inputs.Length != layersSizes[0])
        {
            throw new ArgumentException("Inputs length must be equal to the first layer size");
        }
        
        var result = Feedforward(new Matrix(inputs));

        List<double> resultList = new List<double>();
        
        for (int i = 0; i < result.RowsAmount; i++)
        {
            resultList.Add(result.Values[i, 0]);
        }

        return resultList.ToArray();
    }

    private double PerformLearningIteration(Matrix[] dataSamples, Matrix[] expectedResults)
    {
        Matrix[] changesForWeightsSum = new Matrix[layersAmount - 1];
        Matrix[] changesForBiasesSum = new Matrix[layersAmount - 1];
        for (int i = 0; i < layersAmount - 1; i++)
        {
            changesForWeightsSum[i] = new Matrix(weightsForLayers[i].RowsAmount, weightsForLayers[i].ColumnsAmount);
            changesForBiasesSum[i] = new Matrix(biasesForLayers[i].RowsAmount, biasesForLayers[i].ColumnsAmount);
        }

        double errorSum = 0;

        for (int i = 0; i < dataSamples.Length; i++)
        {
            Matrix prediction = Feedforward(dataSamples[i]);

            var changes = Backpropagation(expectedResults[i], prediction);

            errorSum += CalculateCrossEntropyCost(expectedResults[i], prediction);

            for (int j = 0; j < layersAmount - 1; j++)
            {
                changesForWeightsSum[j] = changesForWeightsSum[j].ElementwiseAdd(changes.changeForWeights[j]);
                changesForBiasesSum[j] = changesForBiasesSum[j].ElementwiseAdd(changes.changeForBiases[j]);
            }
        }

        for (int i = 0; i < layersAmount - 1; i++)
        {
            weightsForLayers[i] = weightsForLayers[i].ElementwiseAdd(changesForWeightsSum[i].ApplyFunction(x => x / dataSamples.Length));
            biasesForLayers[i] = biasesForLayers[i].ElementwiseAdd(changesForBiasesSum[i].ApplyFunction(x => x / dataSamples.Length));
        }

        return errorSum / dataSamples.Length;
    }

    private Matrix Feedforward(Matrix input)
    {
        layersBeforeActivation[0] = input;
        Matrix currentLayer = layersBeforeActivation[0];

        for (int i = 0; i < layersAmount - 1; i++)
        {
            Matrix multipliedByWeightsLayer = Matrix.DotProductMatrices(weightsForLayers[i], currentLayer);

            Matrix layerWithAddedBiases = multipliedByWeightsLayer.ElementwiseAdd(biasesForLayers[i]);
            
            Matrix activatedLayer = activationFunctions[i] switch
            {
                ActivationFunction.ReLU => ReLU(layerWithAddedBiases),
                ActivationFunction.Sigmoid => Sigmoid(layerWithAddedBiases),
                ActivationFunction.Softmax => Softmax(layerWithAddedBiases),
                _ => throw new NotImplementedException()
            };

            layersBeforeActivation[i + 1] = layerWithAddedBiases;
            currentLayer = activatedLayer;
        }

        return currentLayer;
    }

    private (Matrix[] changeForWeights, Matrix[] changeForBiases) Backpropagation(Matrix expectedResults, Matrix predictions)
    {
        Matrix[] changeForWeights = new Matrix[layersAmount - 1];
        Matrix[] changeForBiases = new Matrix[layersAmount - 1];


        Matrix errorMatrix = expectedResults.ElementwiseSubtract(predictions);


        for (int i = layersAmount - 2; i >= 0; i--) // -2 or -1 ??
        {
            Matrix activationDerivativeLayer = activationFunctions[i] switch
            { 
                ActivationFunction.ReLU => DerivativeReLU(layersBeforeActivation[i + 1]),
                ActivationFunction.Sigmoid => DerivativeSigmoid(layersBeforeActivation[i + 1]),
                ActivationFunction.Softmax => DerivativeSoftmax(layersBeforeActivation[i + 1]),
                _ => throw new NotImplementedException()
            };

            Matrix gradientMatrix = activationDerivativeLayer.ElementwiseMultiply(errorMatrix).ApplyFunction(x => x * learningRate);

            Matrix deltaWeightsMatrix = Matrix.DotProductMatrices(gradientMatrix, layersBeforeActivation[i].Transpose());

            changeForWeights[i] = deltaWeightsMatrix;
            changeForBiases[i] = gradientMatrix;

            errorMatrix = Matrix.DotProductMatrices(weightsForLayers[i].Transpose(), errorMatrix);
        }

        return (changeForWeights, changeForBiases);
    }

    


    private double CalculateCrossEntropyCost(Matrix expected, Matrix predictions)
    {
        if(predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        double sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += expected.Values[i, j] * Math.Log(predictions.Values[i, j]);
            }
        }

        return -sum;
    }

    private Matrix ReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x > 0 ? x : 0; });
    }

    private Matrix DerivativeReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x >= 0 ? 1.0 : 0.0; }); 
    }

    private Matrix Sigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x => 1 / (1 + Math.Exp(-x)) );
    }

    private Matrix DerivativeSigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x => {
            var sig = 1 / (1 + Math.Exp(-x));
            return sig * (1 - sig);
        });
    }

    private Matrix Softmax(Matrix mat)
    {
        var expMat = mat.ApplyFunction(x => Math.Exp(x));
        double sumOfMatrix = expMat.Sum();
        return expMat.ApplyFunction(x => x / sumOfMatrix);
    }

    private Matrix DerivativeSoftmax(Matrix mat)
    {
        return Softmax(mat).ApplyFunction(x => x * (1 - x));
    }

}


public enum ActivationFunction
{
    ReLU,
    Sigmoid,
    Softmax,
}