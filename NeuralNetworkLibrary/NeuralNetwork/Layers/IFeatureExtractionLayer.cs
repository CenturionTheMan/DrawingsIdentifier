namespace NeuralNetworkLibrary;

public interface IFeatureExtractionLayer
{
    internal (int outputDepth, int outputHeight, int outputWidth) Initialize((int inputDepth, int inputHeight, int inputWidth) inputShape);


    /// <summary>
    /// Forward pass for the layer
    /// </summary>
    /// <param name="inputs">Data from previous layer</param>
    /// <returns>
    /// output of the layer and other output (max indices in pooling layer, not activated output in
    /// convolution layer)
    /// </returns>
    internal (Matrix[] output, Matrix[] otherOutput) Forward(Matrix[] inputs);


    /// <summary>
    /// Backward pass for the layer
    /// </summary>
    /// <param name="prevOutput">deltas propagated from nextLayer (previous in chain of backward prop)</param>
    /// <param name="currentLayerOutputOther">
    /// output of the current layer. In convolution layer it is
    /// not activated output in pooling layer is ts max indices indexes
    /// </param>
    /// <param name="prevLayerOutputOther">
    /// output of the previous layer (in backprop chain), not activated.
    /// </param>
    internal Matrix[] Backward(Matrix[] prevOutput, Matrix[] prevLayerOutputOther, Matrix[] currentLayerOutputOther, double learningRate);

    internal void UpdateWeightsAndBiases(double batchSize);
}