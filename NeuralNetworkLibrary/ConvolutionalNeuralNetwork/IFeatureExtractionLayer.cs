namespace NeuralNetworkLibrary;

public interface IFeatureExtractionLayer
{
    internal Matrix[] Forward(Matrix[] inputs);
    internal Matrix[] Backward(Matrix[] deltas, Matrix[] previousLayerOutputs, double learningRate);

    internal void UpdateWeightsAndBiases(double batchSize);
}