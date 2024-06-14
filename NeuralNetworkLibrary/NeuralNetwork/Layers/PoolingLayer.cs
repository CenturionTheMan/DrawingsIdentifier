using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord;
using static NeuralNetworkLibrary.ActivationFunctionsHandler;

namespace NeuralNetworkLibrary;

public class PoolingLayer : IFeatureExtractionLayer
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

    (int outputDepth, int outputHeight, int outputWidth) IFeatureExtractionLayer.Initialize((int inputDepth, int inputHeight, int inputWidth) inputShape)
    {
        var size = MatrixExtender.GetSizeAfterPooling((inputShape.inputHeight, inputShape.inputWidth), poolSize, stride);
        return (inputShape.inputDepth, size.outputRows, size.outputColumns);
    }


    void IFeatureExtractionLayer.UpdateWeightsAndBiases(double batchSize)
    {
        
    }

    (Matrix[] output, Matrix[] otherOutput) IFeatureExtractionLayer.Forward(Matrix[] inputs)
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

    Matrix[] IFeatureExtractionLayer.Backward(Matrix[] deltas, Matrix[] prevLayerSize, Matrix[] maxIndexMap, double learningRate)
    {
        Matrix[] result = new Matrix[deltas.Length];
        for (int i = 0; i < deltas.Length; i++)
        {
            result[i] = new Matrix(prevLayerSize[i].RowsAmount, prevLayerSize[i].ColumnsAmount);

            for (int j = 0; j < deltas[i].RowsAmount; j++)
            {
                for (int k = 0; k < deltas[i].ColumnsAmount; k++)
                {
                    int maxIndex = (int)maxIndexMap[i][j, k];
                    
                    var dim = IndexCalculations.GetRowAndColumn(maxIndex, prevLayerSize[i].ColumnsAmount);
                    result[i][dim.row, dim.column] += deltas[i][j, k];
                }
            }
        }
        return result;
    }
}