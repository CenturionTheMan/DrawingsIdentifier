namespace NeuralNetworkLibrary;

public class FullyConnectedLayer
{
    public readonly int LayerSize;
    public readonly ActivationFunction ActivationFunction;

    private Matrix weights;
    private Matrix biases;

    private Matrix weightsGradientSum;
    private Matrix biasesGradientSum;

    private int previousLayerSize;

    private double minWeight;
    private double maxWeight;

    public FullyConnectedLayer(int layerSize, ActivationFunction ActivationFunction, double minWeight = -0.2, double maxWeight = 0.2)
    {
        this.LayerSize = layerSize;
        this.ActivationFunction = ActivationFunction;
        this.minWeight = minWeight;
        this.maxWeight = maxWeight;

        this.previousLayerSize = -1;
        this.weights = new Matrix(0, 0);
        this.biases = new Matrix(0, 0);
        this.weightsGradientSum = new Matrix(0, 0);
        this.biasesGradientSum = new Matrix(0, 0);
    }

    internal void Initialize(int previousLayerSize)
    {
        this.previousLayerSize = previousLayerSize;

        this.weights = new Matrix(LayerSize, previousLayerSize, minWeight, maxWeight);
        this.biases = new Matrix(LayerSize, 1, minWeight, maxWeight);

        this.weightsGradientSum = new Matrix(LayerSize, previousLayerSize);
        this.biasesGradientSum = new Matrix(LayerSize, 1);
    }

    internal (Matrix activatedOutput, Matrix outputBeforeActivation) Forward(Matrix input, Matrix prevLayerOutputBeforeActivation)
    {
        //this.prevLayerOutputBeforeActivation = prevLayerOutputBeforeActivation;

        Matrix currentLayer = input;

        Matrix multipliedByWeightsLayer = Matrix.DotProductMatrices(weights, currentLayer);
        Matrix layerWithAddedBiases = multipliedByWeightsLayer.ElementWiseAdd(biases);
        //layerOutputBeforeActivation = layerWithAddedBiases;

        Matrix activatedLayer = ActivationFunction switch
        {
            ActivationFunction.ReLU => ActivationFunctionsHandler.ReLU(layerWithAddedBiases),
            ActivationFunction.Sigmoid => ActivationFunctionsHandler.Sigmoid(layerWithAddedBiases),
            ActivationFunction.Softmax => ActivationFunctionsHandler.Softmax(layerWithAddedBiases),
            _ => throw new NotImplementedException()
        };

        return (activatedLayer, layerWithAddedBiases);
    }

    internal Matrix Backward(Matrix errorMatrix, Matrix prevLayerOutputBeforeActivation, Matrix thisLayerOutputBeforeActivation, double learningRate)
    {
        Matrix activationDerivativeLayer = ActivationFunction switch
        {
            ActivationFunction.ReLU => ActivationFunctionsHandler.DerivativeReLU(thisLayerOutputBeforeActivation),
            ActivationFunction.Sigmoid => ActivationFunctionsHandler.DerivativeSigmoid(thisLayerOutputBeforeActivation),
            ActivationFunction.Softmax => ActivationFunctionsHandler.DerivativeSoftmax(thisLayerOutputBeforeActivation),
            _ => throw new NotImplementedException()
        };

        Matrix gradientMatrix = activationDerivativeLayer.ElementWiseMultiply(errorMatrix).ApplyFunction(x => x * learningRate);
        Matrix deltaWeightsMatrix = Matrix.DotProductMatrices(gradientMatrix, prevLayerOutputBeforeActivation.Transpose());

        weightsGradientSum = weightsGradientSum.ElementWiseAdd(deltaWeightsMatrix);
        biasesGradientSum = biasesGradientSum.ElementWiseAdd(gradientMatrix);

        return Matrix.DotProductMatrices(weights.Transpose(), errorMatrix);
    }

    internal void UpdateWeightsAndBiases(int batchSize)
    {
        weights = weights.ElementWiseAdd(weightsGradientSum.ApplyFunction(x => x / batchSize));
        biases = biases.ElementWiseAdd(biasesGradientSum.ApplyFunction(x => x / batchSize));

        weightsGradientSum = new Matrix(LayerSize, weights.ColumnsAmount);
        biasesGradientSum = new Matrix(LayerSize, 1);
    }
}