namespace NeuralNetworkLibrary;

public interface IFeatureExtractionLayer
{
    public int Depth { get; }
    public int KernelSize { get; }
    public ActivationFunction ActivationFunction { get; }
    public int Stride { get; }

    internal void Initialize((int inputDepth, int inputHeight, int inputWidth) inputShape);

    internal (Matrix[] output, Matrix[] outputsBeforeActivation) Forward(Matrix[] inputs);

    internal Matrix[] Backward(Matrix[] deltas, Matrix[] previousLayerOutputs, double learningRate);

    internal void UpdateWeightsAndBiases(double batchSize);
}