using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

internal static class Utilities
{
    public static Matrix FlattenMatrix(Matrix[] matrices)
    {
        List<double> flattenedList = new();
        foreach (var matrix in matrices)
        {
           for (int i = 0; i < matrix.RowsAmount; i++)
            {
                for (int j = 0; j < matrix.ColumnsAmount; j++)
                {
                    flattenedList.Add(matrix[i, j]);
                }
            }
        }

        return new Matrix(flattenedList.ToArray());
    }

    public static Matrix[] UnflattenMatrix(Matrix flattenedMatrix, int matrixSize)
    {
        if(flattenedMatrix.RowsAmount % (matrixSize * matrixSize) != 0)
            throw new ArgumentException("Invalid matrix size");

        List<Matrix> matrices = new List<Matrix>();

        int index = 0;
        for(int i = 0; i < flattenedMatrix.RowsAmount; i+= matrixSize * matrixSize)
        {
            Matrix matrix = new Matrix(matrixSize, matrixSize);
            for (int j = 0; j < matrixSize; j++)
            {
                for (int k = 0; k < matrixSize; k++)
                {
                    matrix[j, k] = flattenedMatrix[index++, 0];
                }
            }
            matrices.Add(matrix);
        }


        return matrices.ToArray();
    }

    public static Matrix ApplyActivationFunction(this Matrix input, ActivationFunction activationFunction)
    {
        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                return Utilities.ReLU(input);
            case ActivationFunction.Sigmoid:
                return Utilities.Sigmoid(input);
            case ActivationFunction.Softmax:
                return Utilities.Softmax(input);
            default:
                throw new ArgumentException("Invalid activation function");
        }
    }

    public static Matrix DerivativeActivationFunction(this Matrix input, ActivationFunction activationFunction)
    {
        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                return Utilities.DerivativeReLU(input);
            case ActivationFunction.Sigmoid:
                return Utilities.DerivativeSigmoid(input);
            case ActivationFunction.Softmax:
                return Utilities.DerivativeSoftmax(input);
            default:
                throw new ArgumentException("Invalid activation function");
        }
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
                sum += Math.Pow(expected[i, j] - predictions[i, j], 2);
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
                sum += expected[i, j] * Math.Log(predictions[i, j]);
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