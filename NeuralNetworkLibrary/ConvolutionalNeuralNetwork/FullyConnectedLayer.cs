namespace NeuralNetworkLibrary;

public class FullyConnectedLayer
{
    public readonly int LayerSize;
    public readonly ActivationFunction ActivationFunction;

    internal bool IsInitialized {get => isInitialized; }

    private bool isInitialized = false;
    private Matrix weights;
    private Matrix biases;

    // private Matrix prevLayerOutputBeforeActivation;
    // private Matrix layerOutputBeforeActivation;

    private Matrix weightsGradientSum;
    private Matrix biasesGradientSum;

    public FullyConnectedLayer(int layerSize, ActivationFunction ActivationFunction) 
    {
        this.LayerSize = layerSize;
        this.ActivationFunction = ActivationFunction;

        this.weights = new Matrix(0,0);
        this.biases = new Matrix(0,0);
        // this.layerOutputBeforeActivation = new Matrix(0,0);
        // this.prevLayerOutputBeforeActivation = new Matrix(0,0);

        this.isInitialized = false;
    }
    
    internal void InitializeWeightsAndBiases(int previousLayerSize, double minWeight, double maxWeight)
    {
        this.weights = new Matrix(LayerSize, previousLayerSize, minWeight, maxWeight);
        this.biases = new Matrix(LayerSize, 1, minWeight, maxWeight);

        this.weightsGradientSum = new Matrix(LayerSize, previousLayerSize);
        this.biasesGradientSum = new Matrix(LayerSize, 1);

        this.isInitialized = true;
    }

    internal (Matrix activatedOutput, Matrix outputBeforeActivation) Forward(Matrix input, Matrix prevLayerOutputBeforeActivation)
    {
        if(!isInitialized)
            throw new Exception("Layer is not initialized");

        //this.prevLayerOutputBeforeActivation = prevLayerOutputBeforeActivation;

        Matrix currentLayer = input;

        Matrix multipliedByWeightsLayer = Matrix.DotProductMatrices(weights, currentLayer);
        Matrix layerWithAddedBiases = multipliedByWeightsLayer.ElementWiseAdd(biases);
        //layerOutputBeforeActivation = layerWithAddedBiases;

        Matrix activatedLayer = ActivationFunction switch
        {
            ActivationFunction.ReLU => Utilities.ReLU(layerWithAddedBiases),
            ActivationFunction.Sigmoid => Utilities.Sigmoid(layerWithAddedBiases),
            ActivationFunction.Softmax => Utilities.Softmax(layerWithAddedBiases),
            _ => throw new NotImplementedException()
        };

        return (activatedLayer, layerWithAddedBiases);
    }

    internal Matrix Backward(Matrix errorMatrix, Matrix prevLayerOutputBeforeActivation, Matrix thisLayerOutputBeforeActivation, double learningRate)
    {
        if(!isInitialized)
            throw new Exception("Layer is not initialized");

        Matrix activationDerivativeLayer = ActivationFunction switch
        {
            ActivationFunction.ReLU => Utilities.DerivativeReLU(thisLayerOutputBeforeActivation),
            ActivationFunction.Sigmoid => Utilities.DerivativeSigmoid(thisLayerOutputBeforeActivation),
            ActivationFunction.Softmax => Utilities.DerivativeSoftmax(thisLayerOutputBeforeActivation),
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
        if(!isInitialized)
            throw new Exception("Layer is not initialized");

        weights = weights.ElementWiseAdd(weightsGradientSum.ApplyFunction(x => x / batchSize));
        biases = biases.ElementWiseAdd(biasesGradientSum.ApplyFunction(x => x / batchSize));

        weightsGradientSum = new Matrix(LayerSize, weights.ColumnsAmount);
        biasesGradientSum = new Matrix(LayerSize, 1);
    }
}