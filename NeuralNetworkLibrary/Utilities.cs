using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

internal static class Utilities
{
    public static (Matrix flattened, IEnumerable<(int, int)> dimensions) FlattenMatrices(Matrix[] matrices)
    {
        List<(int, int)> dimensions = new List<(int, int)>();
        List<double> values = new List<double>();

        foreach (var matrix in matrices)
        {
            dimensions.Add((matrix.RowsAmount, matrix.ColumnsAmount));
            foreach (var value in matrix)
            {
                values.Add(value);
            }
        }

        Matrix flattenedMatrix = new Matrix(values.ToArray());
        return (flattenedMatrix, dimensions);
    }

    public static Matrix[] RecreateMatrices(Matrix flattenedMatrix, IEnumerable<(int, int)> dimensions)
    {
        List<Matrix> matrices = new List<Matrix>();
        int currentRow = 0;

        foreach (var (rowCount, columnCount) in dimensions)
        {
            double[,] matrixValues = new double[rowCount, columnCount];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    matrixValues[i, j] = flattenedMatrix[currentRow, 0];
                    currentRow++;
                }
            }

            Matrix matrix = new Matrix(matrixValues);
            matrices.Add(matrix);
        }

        return matrices.ToArray();
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