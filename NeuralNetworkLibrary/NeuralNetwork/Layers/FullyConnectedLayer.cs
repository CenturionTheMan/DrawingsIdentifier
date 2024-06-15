using System.Data.Common;

namespace NeuralNetworkLibrary;

public class FullyConnectedLayer : ILayer
{
    LayerType ILayer.LayerType => LayerType.FullyConnected;

    private const double maxNorm = 0.5;

    private ActivationFunction activationFunction;
    private int layerSize;

    private Matrix weights;
    private Matrix biases;

    private Matrix weightsGradientSum;
    private Matrix biasesGradientSum;

    public FullyConnectedLayer(int previousLayerSize, int layerSize, ActivationFunction activationFunction) 
    {
        this.weights = new Matrix(layerSize, previousLayerSize);
        this.biases = new Matrix(layerSize, 1);

        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                this.weights.InitializeHe();
                break;

            case ActivationFunction.Sigmoid:
                this.weights.InitializeXavier();
                break;

            case ActivationFunction.Softmax:
                this.weights.InitializeXavier();
                break;
            default:
                throw new NotImplementedException();
        }

        this.weightsGradientSum = new Matrix(layerSize, previousLayerSize);
        this.biasesGradientSum = new Matrix(layerSize, 1);

        this.activationFunction = activationFunction;
        this.layerSize = layerSize;
    }
    
    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] input)
    {
        if(input.Length != 1)
            throw new ArgumentException("Fully connected layer can only have one input");

        Matrix currentLayer = input[0];

        Matrix multipliedByWeightsLayer = Matrix.DotProductMatrices(weights, currentLayer);
        Matrix layerWithAddedBiases = multipliedByWeightsLayer.ElementWiseAdd(biases);

        Matrix activatedLayer = activationFunction switch
        {
            ActivationFunction.ReLU => ActivationFunctionsHandler.ReLU(layerWithAddedBiases),
            ActivationFunction.Sigmoid => ActivationFunctionsHandler.Sigmoid(layerWithAddedBiases),
            ActivationFunction.Softmax => ActivationFunctionsHandler.Softmax(layerWithAddedBiases),
            _ => throw new NotImplementedException()
        };

        return ([activatedLayer], [layerWithAddedBiases]);
    }

    Matrix[] ILayer.Backward(Matrix[] errorMatrix, Matrix[] prevLayerOutputBeforeActivation, Matrix[] thisLayerOutputBeforeActivation, double learningRate)
    {
        if(errorMatrix.Length != 1)
            throw new ArgumentException("Fully connected layer can only have one input");

        Matrix activationDerivativeLayer = activationFunction switch
        {
            ActivationFunction.ReLU => ActivationFunctionsHandler.DerivativeReLU(thisLayerOutputBeforeActivation[0]),
            ActivationFunction.Sigmoid => ActivationFunctionsHandler.DerivativeSigmoid(thisLayerOutputBeforeActivation[0]),
            ActivationFunction.Softmax => ActivationFunctionsHandler.DerivativeSoftmax(thisLayerOutputBeforeActivation[0]),
            _ => throw new NotImplementedException()
        };

        Matrix gradientMatrix = activationDerivativeLayer.ElementWiseMultiply(errorMatrix[0]).ApplyFunction(x => x * learningRate);
        Matrix deltaWeightsMatrix = Matrix.DotProductMatrices(gradientMatrix, prevLayerOutputBeforeActivation[0].Transpose());

        weightsGradientSum = weightsGradientSum.ElementWiseAdd(deltaWeightsMatrix);
        biasesGradientSum = biasesGradientSum.ElementWiseAdd(gradientMatrix);

        return [Matrix.DotProductMatrices(weights.Transpose(), errorMatrix[0])];
    }

    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        double multiplier = 1.0 / (double)batchSize;
        var changeForWeights = weightsGradientSum * multiplier;

        double clipCoefficient = maxNorm / (changeForWeights.GetNorm() + double.Epsilon);
        if (clipCoefficient < 1)
            changeForWeights = changeForWeights * clipCoefficient;

        weights = weights.ElementWiseAdd(changeForWeights);
        biases = biases.ElementWiseAdd(biasesGradientSum.ApplyFunction(x => x * multiplier));

        weightsGradientSum = new Matrix(layerSize, weights.ColumnsAmount);
        biasesGradientSum = new Matrix(layerSize, 1);
    }
}