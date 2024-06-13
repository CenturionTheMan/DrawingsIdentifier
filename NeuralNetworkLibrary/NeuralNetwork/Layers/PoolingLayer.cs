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
            (result[i], maxIndexMap[i]) = MatrixExtender.MaxPooling(inputs[i], poolSize, stride);
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

    
}