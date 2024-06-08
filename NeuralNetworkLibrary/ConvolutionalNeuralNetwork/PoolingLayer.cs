using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NeuralNetworkLibrary.Utilities;

namespace NeuralNetworkLibrary;

public class PoolingLayer : IFeatureExtractionLayer
{
    private int poolSize;
    private int stride;
    private int[][]? maxIndices;

    // private Matrix[] previousLayerOutputs;

    public PoolingLayer(int poolSize, int stride)
    {
        this.poolSize = poolSize;
        this.stride = stride;
        this.maxIndices = null;

        // this.previousLayerOutputs = new Matrix[0];
    }

    Matrix[] IFeatureExtractionLayer.Forward(Matrix[] inputs)
    {
        // this.previousLayerOutputs = inputs;

        Matrix[] result = new Matrix[inputs.Length];
        maxIndices = new int[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
        {
            (result[i], maxIndices[i]) = MaxPooling(inputs[i], poolSize, stride);
        }
        return result;
    }

    Matrix[] IFeatureExtractionLayer.Backward(Matrix[] deltas, Matrix[] previousLayerOutputs, double learningRate)
    {
        if(maxIndices == null)
        {
            throw new InvalidOperationException("Forward method must be called before calling Backward method");
        }

        Matrix[] result = new Matrix[deltas.Length];
        for (int i = 0; i < deltas.Length; i++)
        {
            result[i] = new Matrix(previousLayerOutputs[i].RowsAmount, previousLayerOutputs[i].ColumnsAmount);
            for (int j = 0; j < maxIndices[i].Length; j++)
            {
                int row = maxIndices[i][j] / previousLayerOutputs[i].ColumnsAmount;
                int col = maxIndices[i][j] % previousLayerOutputs[i].ColumnsAmount;
                result[i][row, col] += deltas[i][j / deltas[i].ColumnsAmount, j % deltas[i].ColumnsAmount];
            }
        }
        return result;
    }

    private (Matrix, int[]) MaxPooling(Matrix matrix, int poolSize, int stride)
    {
        int newRows = (matrix.RowsAmount - poolSize) / stride + 1;
        int newColumns = (matrix.ColumnsAmount - poolSize) / stride + 1;
        Matrix result = new Matrix(newRows, newColumns);
        List<int> indices = new();

        for (int i = 0; i < newRows; i++)
        {
            for (int j = 0; j < newColumns; j++)
            {
                double max = double.MinValue;
                int maxIndex = -1;
                for (int x = 0; x < poolSize; x++)
                {
                    for (int y = 0; y < poolSize; y++)
                    {
                        int rowIndex = i * stride + x;
                        int colIndex = j * stride + y;
                        if (rowIndex < matrix.RowsAmount && colIndex < matrix.ColumnsAmount && matrix[rowIndex, colIndex] > max)
                        {
                            max = matrix[rowIndex, colIndex];
                            maxIndex = rowIndex * matrix.ColumnsAmount + colIndex;
                        }
                    }
                }
                result[i, j] = max;
                indices.Add(maxIndex);
            }
        }

        return (result, indices.ToArray());
    }

    void IFeatureExtractionLayer.UpdateWeightsAndBiases(double batchSize)
    {
        
    }
}