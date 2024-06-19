using System.Xml;
using System.Xml.Linq;

namespace NeuralNetworkLibrary;

internal class ReshapeFeatureToClassificationLayer : ILayer
{
    #region PARAMS

    LayerType ILayer.LayerType => LayerType.Reshape;

    private int rowsAmount;
    private int columnsAmount;

    #endregion PARAMS

    #region CTOR

    public ReshapeFeatureToClassificationLayer(int rowsAmount, int columnsAmount)
    {
        this.rowsAmount = rowsAmount;
        this.columnsAmount = columnsAmount;
    }

    internal static ILayer? LoadLayerData(XElement layerHead, XElement layerData)
    {
        string? rowsAmount = layerHead.Element("RowsAmount")?.Value;
        string? columnsAmount = layerHead.Element("ColumnsAmount")?.Value;

        if (rowsAmount == null || columnsAmount == null)
            return null;

        return new ReshapeFeatureToClassificationLayer(int.Parse(rowsAmount), int.Parse(columnsAmount));
    }

    #endregion CTOR

    #region METHODS

    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        var flattenedMatrix = MatrixExtender.FlattenMatrix(inputs);
        return ([flattenedMatrix], [flattenedMatrix]);
    }

    Matrix[] ILayer.Backward(Matrix[] prevOutput, Matrix[] prevLayerOutputOther, Matrix[] currentLayerOutputOther, double learningRate)
    {
        if (prevOutput.Length != 1)
            throw new ArgumentException("Reshape layer can only have one input");

        Matrix[] errorMatrices = MatrixExtender.UnflattenMatrix(prevOutput[0], rowsAmount, columnsAmount);
        return errorMatrices;
    }

    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        //nothing to do here
    }

    #endregion METHODS

    #region SAVE

    void ILayer.SaveLayerDescription(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerHead");
        doc.WriteAttributeString("LayerType", $"{LayerType.Reshape.ToString()}");
        doc.WriteElementString("RowsAmount", rowsAmount.ToString());
        doc.WriteElementString("ColumnsAmount", columnsAmount.ToString());
        doc.WriteEndElement();
    }

    void ILayer.SaveLayerData(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerData");
        doc.WriteEndElement();
    }

    #endregion SAVE
}