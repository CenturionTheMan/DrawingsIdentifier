namespace NeuralNetworkLibrary;

class ReshapeFeatureToClassificationLayer : ILayer
{
    LayerType ILayer.LayerType => LayerType.Reshape;

    private int rowsAmount;
    private int columnsAmount;

    public ReshapeFeatureToClassificationLayer(int rowsAmount, int columnsAmount)
    {
        this.rowsAmount = rowsAmount;
        this.columnsAmount = columnsAmount;
    }

    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        var flattenedMatrix = MatrixExtender.FlattenMatrix(inputs);
        return ([flattenedMatrix], [flattenedMatrix]);
    }

    Matrix[] ILayer.Backward(Matrix[] prevOutput, Matrix[] prevLayerOutputOther, Matrix[] currentLayerOutputOther, double learningRate)
    {
        if(prevOutput.Length != 1)
            throw new ArgumentException("Reshape layer can only have one input");

        Matrix[] errorMatrices = MatrixExtender.UnflattenMatrix(prevOutput[0], rowsAmount, columnsAmount);
        return errorMatrices;
    }

    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        //nothing to do here
    }
}