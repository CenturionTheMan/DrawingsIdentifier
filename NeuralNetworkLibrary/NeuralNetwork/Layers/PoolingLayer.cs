using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NeuralNetworkLibrary.ActivationFunctionsHandler;

namespace NeuralNetworkLibrary;

public class PoolingLayer 
{
    public int PoolSize => poolSize;
    public int Stride => stride;

    private int poolSize;
    private int stride;

    public PoolingLayer(int poolSize, int stride)
    {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    (Matrix[] output, Matrix[] maxIndexMap) Forward(Matrix[] inputs)
    {
        // this.previousLayerOutputs = inputs;

        Matrix[] result = new Matrix[inputs.Length];
        Matrix[] maxIndexMap = new Matrix[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            (result[i], maxIndexMap[i]) = MaxPooling(inputs[i], poolSize, stride);
        }
        return (result, maxIndexMap);
    }

    Matrix[] Backward(Matrix[] deltas, Matrix[] maxIndexMap, Matrix[] previousLayerOutputs)
    {
        Matrix[] result = new Matrix[deltas.Length];
        for (int i = 0; i < deltas.Length; i++)
        {
            result[i] = new Matrix(previousLayerOutputs[i].RowsAmount, previousLayerOutputs[i].ColumnsAmount);

            for (int j = 0; j < deltas[i].RowsAmount; j++)
            {
                for (int k = 0; k < deltas[i].ColumnsAmount; k++)
                {
                    int maxIndex = (int)maxIndexMap[i][j, k];
                    int rowIndex = maxIndex / previousLayerOutputs[i].ColumnsAmount;
                    int colIndex = maxIndex % previousLayerOutputs[i].ColumnsAmount;
                    result[i][rowIndex, colIndex] += deltas[i][j, k];
                }
            }
        }
        return result;
    }

    private (Matrix, Matrix) MaxPooling(Matrix matrix, int poolSize, int stride)
    {
        int newRows = (matrix.RowsAmount - poolSize) / stride + 1;
        int newColumns = (matrix.ColumnsAmount - poolSize) / stride + 1;
        Matrix result = new Matrix(newRows, newColumns);
        Matrix maxIndexMap = new Matrix(newRows, newColumns);

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
                maxIndexMap[i, j] = maxIndex;
            }
        }

        return (result, maxIndexMap);
    }
}