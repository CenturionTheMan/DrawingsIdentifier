using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

internal static class Utilities
{
    /// <summary>
    /// Takes selected slice from the original matrix
    /// </summary>
    /// <param name="original">Original matrix</param>
    /// <param name="startX">start pos X</param>
    /// <param name="startY">Start pos Y</param>
    /// <param name="sliceSizeX">Slice to cut x</param>
    /// <param name="sliceSizeY">Slice to cut Y</param>
    /// <returns>Slice from original matrix</returns>
    public static Matrix GetSlice(Matrix original, int startX, int startY, int sliceSizeX, int sliceSizeY)
    {
        return GetSlice(original.Values, startX, startY, sliceSizeX, sliceSizeY);
    }

    /// <summary>
    /// Takes selected slice from the original matrix
    /// </summary>
    /// <param name="original">Original values</param>
    /// <param name="startX">start pos X</param>
    /// <param name="startY">Start pos Y</param>
    /// <param name="sliceSizeX">Slice to cut x</param>
    /// <param name="sliceSizeY">Slice to cut Y</param>
    /// <returns>Slice from original matrix</returns>
    public static Matrix GetSlice(double[,] original, int startX, int startY, int sliceSizeX, int sliceSizeY)
    {
        double[,] slice = new double[sliceSizeX, sliceSizeY];

        for (int i = 0; i < sliceSizeX; i++)
        {
            for (int j = 0; j < sliceSizeY; j++)
            {
                slice[i, j] = original[startX + i, startY + j];
            }
        }

        return new Matrix(slice);
    }

    #region Activation Functions and Error

    /// <summary>
    /// Calculates the mean squared error between the expected and predicted results.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    internal static double CalculateMeanSquaredError(Matrix expected, Matrix predictions)
    {
        if (predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        double sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += Math.Pow(expected.Values[i, j] - predictions.Values[i, j], 2);
            }
        }

        return sum / (predictions.RowsAmount * predictions.ColumnsAmount);
    }

    /// <summary>
    /// Calculates the cross entropy cost between the expected and predicted results.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    internal static double CalculateCrossEntropyCost(Matrix expected, Matrix predictions)
    {
        if (predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        double sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += expected.Values[i, j] * Math.Log(predictions.Values[i, j]);
            }
        }

        return -sum;
    }

    /// <summary>
    /// Applies the ReLU activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    internal static Matrix ReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x > 0 ? x : 0; });
    }

    /// <summary>
    /// Applies the derivative of the ReLU activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    internal static Matrix DerivativeReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x >= 0 ? 1.0 : 0.0; });
    }

    /// <summary>
    /// Applies the Sigmoid activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    internal static Matrix Sigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x => 1 / (1 + Math.Exp(-x)));
    }

    /// <summary>
    /// Applies the derivative of the Sigmoid activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    internal static Matrix DerivativeSigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x =>
        {
            var sig = 1 / (1 + Math.Exp(-x));
            return sig * (1 - sig);
        });
    }

    /// <summary>
    /// Applies the Softmax activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    internal static Matrix Softmax(Matrix mat)
    {
        var expMat = mat.ApplyFunction(x => Math.Exp(x));
        double sumOfMatrix = expMat.Sum();
        return expMat.ApplyFunction(x => x / sumOfMatrix);
    }

    /// <summary>
    /// Applies the derivative of the Softmax activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    internal static Matrix DerivativeSoftmax(Matrix mat)
    {
        return Softmax(mat).ApplyFunction(x => x * (1 - x));
    }

    #endregion Activation Functions and Error
}