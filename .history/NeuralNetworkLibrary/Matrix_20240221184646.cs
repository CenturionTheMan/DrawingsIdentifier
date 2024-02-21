using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;


namespace NeuralNetworkLibrary;

internal class Matrix
{
    private static Random random = new();

    internal double[,] Values { get; set; }
    internal readonly int RowsAmount;
    internal readonly int ColumnsAmount;

    internal Matrix(double[] singleColumnValues)
    {
        this.Values = new double[singleColumnValues.Length,1];

        for (int i = 0; i < singleColumnValues.Length; i++)
        {
            Values[i, 0] = singleColumnValues[i];
        }

        RowsAmount = singleColumnValues.Length;
        ColumnsAmount = 1;
    }

    internal Matrix(double[,] values)
    {
        this.Values = values;
        RowsAmount = values.GetLength(0);
        ColumnsAmount = values.GetLength(1);
    }

    internal Matrix(int rowsAmount, int columnsAmount)
    {
        Values = new double[rowsAmount, columnsAmount];
        RowsAmount = rowsAmount;
        ColumnsAmount = columnsAmount;
    }

    internal Matrix(int rowsAmount, int columnsAmount, double min, double max)
    {
        Values = new double[rowsAmount, columnsAmount];
        RowsAmount = rowsAmount;
        ColumnsAmount = columnsAmount;
        
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                Values[i, j] = random.NextDouble() * (max - min) + min;
            }
        }
    }


    internal static Matrix DotProductMatrices(Matrix a, Matrix b)
    {
        return new Matrix(Accord.Math.Matrix.Dot(a.Values, b.Values));
    }

    internal static Matrix ElementwiseMultiplyMatrices(Matrix a, Matrix b)
    {
        return EachElementAssignment(a, b, (i, j) => a.Values[i, j] * b.Values[i, j]);
    }

    internal static Matrix ElementwiseAddMatrices(Matrix a, Matrix b)
    {
        return EachElementAssignment(a, b, (i, j) => a.Values[i, j] + b.Values[i, j]);
    }

    internal static Matrix ElementwiseSubtractMatrices(Matrix a, Matrix b)
    {
        return EachElementAssignment(a, b, (i, j) => a.Values[i, j] - b.Values[i, j]);
    }


    internal static Matrix EachElementAssignment(Matrix a, Matrix b, Func<int, int, double> mathOperation)
    {
        if (a.RowsAmount != b.RowsAmount || a.ColumnsAmount != b.ColumnsAmount)
        {
            throw new Exception("Matrices must have the same dimensions");
        }

        Matrix result = new Matrix(a.RowsAmount, a.ColumnsAmount);

        Parallel.For(0, a.RowsAmount, i =>
        {
            Parallel.For(0, a.ColumnsAmount, j =>
            {
                result.Values[i, j] = mathOperation(i, j);
            });
        });

        return result;
    }
}

internal static class MatrixExtender
{
    internal static double Sum(this Matrix a)
    {
        double sum = 0;
        foreach (var item in a.Values)
        {
            sum += item;
        }
        return sum;
    }

    internal static Matrix Transpose(this Matrix a)
    {
        Matrix result = new Matrix(a.ColumnsAmount, a.RowsAmount);

        for (int i = 0; i < a.ColumnsAmount; i++)
        {
            for (int j = 0; j < a.RowsAmount; j++)
            {
                result.Values[i,j] = a.Values[j,i];
            }
        }

        return result;
    }

    internal static Matrix ApplyFunction(this Matrix a, Func<double, double> function)
    {
        Matrix result = new Matrix(a.RowsAmount, a.ColumnsAmount);

        for (int i = 0; i < a.RowsAmount; i++)
        {
            for (int j = 0; j < a.ColumnsAmount; j++)
            {
                result.Values[i, j] = function(a.Values[i, j]);
            }
        }
        return result;
    }

    internal static Matrix DotProduct(this Matrix a, Matrix b)
    {
        return Matrix.DotProductMatrices(a, b);
    }

    internal static Matrix ElementwiseMultiply(this Matrix a, Matrix b)
    {
        return Matrix.ElementwiseMultiplyMatrices(a, b);
    }

    internal static Matrix ElementwiseAdd(this Matrix a, Matrix b)
    {
        return Matrix.ElementwiseAddMatrices(a, b);
    }

    internal static Matrix ElementwiseSubtract(this Matrix a, Matrix b)
    {
        return Matrix.ElementwiseSubtractMatrices(a, b);
    }
}
