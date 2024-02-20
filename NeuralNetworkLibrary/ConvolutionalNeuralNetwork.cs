namespace NeuralNetworkLibrary;


public class ConvolutionalNeuralNetwork
{
    private int[] layersSizes;
    private Matrix[] biasesForLayers;
    private Matrix[] weightsForLayers;
    private Matrix[] layersBeforeActivation;
    private double learningRate;
    private ActivationFunction[] activationFunctions;
    private int layersAmount;

    public ConvolutionalNeuralNetwork(int[] layersSizes, ActivationFunction[] activationFunctions)
    {
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



    private Matrix Feedforward(Matrix input)
    {
        layersBeforeActivation[0] = input;
        Matrix currentLayer = layersBeforeActivation[0];

        for (int i = 0; i < layersAmount - 1; i++)
        {
            Matrix multipliedByWeightsLayer = MatrixHandler.DotProductMatrices(weightsForLayers[i], currentLayer);

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

            Matrix deltaWeightsMatrix = MatrixHandler.DotProductMatrices(gradientMatrix, layersBeforeActivation[i].Transpose());

            changeForWeights[i] = deltaWeightsMatrix;
            changeForBiases[i] = gradientMatrix;

            errorMatrix = MatrixHandler.DotProductMatrices(weightsForLayers[i].Transpose(), errorMatrix);
        }

        return (changeForWeights, changeForBiases);
    }





    private Matrix ReLU(Matrix mat)
    {
        if(mat.ColumnsAmount != 1)
            throw new Exception($"Matrix have to represent layer! Must have single column and was {mat.ColumnsAmount}.");

        return mat.ApplyFunction(x => { return x > 0 ? x : 0; });
    }

    private Matrix DerivativeReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x >= 0 ? 1.0 : 0.0; }); 
    }

    private Matrix Sigmoid(Matrix mat)
    {
        if (mat.ColumnsAmount != 1)
            throw new Exception($"Matrix have to represent layer! Must have single column and was {mat.ColumnsAmount}.");

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
        if (mat.ColumnsAmount != 1)
            throw new Exception($"Matrix have to represent layer! Must have single column and was {mat.ColumnsAmount}.");

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