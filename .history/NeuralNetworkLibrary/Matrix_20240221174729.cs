using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;


namespace NeuralNetworkLibrary;

public class Matrix
{
    private static Random random = new();

    public double[,] Values { get; internal set; }
    public readonly int RowsAmount;
    public readonly int ColumnsAmount;

    public Matrix(double[] singleColumnValues)
    {
        this.Values = new double[singleColumnValues.Length,1];

        for (int i = 0; i < singleColumnValues.Length; i++)
        {
            Values[i, 0] = singleColumnValues[i];
        }

        RowsAmount = singleColumnValues.Length;
        ColumnsAmount = 1;
    }

    public Matrix(double[,] values)
    {
        this.Values = values;
        RowsAmount = values.GetLength(0);
        ColumnsAmount = values.GetLength(1);
    }

    public Matrix(int rowsAmount, int columnsAmount)
    {
        Values = new double[rowsAmount, columnsAmount];
        RowsAmount = rowsAmount;
        ColumnsAmount = columnsAmount;
    }

    public Matrix(int rowsAmount, int columnsAmount, double min, double max)
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
}

public static class MatrixHandler
{
    public static double Sum(this Matrix a)
    {
        double sum = 0;
        foreach (var item in a.Values)
        {
            sum += item;
        }
        return sum;
    }

    public static Matrix Transpose(this Matrix a)
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

    public static Matrix ApplyFunction(this Matrix a, Func<double, double> function)
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

    public static Matrix DotProduct(this Matrix a, Matrix b)
    {
        return DotProductMatrices(a, b);
    }

    public static Matrix ElementwiseMultiply(this Matrix a, Matrix b)
    {
        return ElementwiseMultiplyMatrices(a, b);
    }

    public static Matrix ElementwiseAdd(this Matrix a, Matrix b)
    {
        return ElementwiseAddMatrices(a, b);
    }

    public static Matrix ElementwiseSubtract(this Matrix a, Matrix b)
    {
        return ElementwiseSubtractMatrices(a, b);
    }




    public static Matrix DotProductMatrices(Matrix a, Matrix b)
    {
        return new Matrix(Accord.Math.Matrix.Dot(a.Values, b.Values));
    }

    public static Matrix ElementwiseMultiplyMatrices(Matrix a, Matrix b)
    {
        return EachElemenAssigment(a, b, (i, j) => a.Values[i, j] * b.Values[i, j]);
    }

    public static Matrix ElementwiseAddMatrices(Matrix a, Matrix b)
    {
        return EachElemenAssigment(a, b, (i, j) => a.Values[i, j] + b.Values[i, j]);
    }

    public static Matrix ElementwiseSubtractMatrices(Matrix a, Matrix b)
    {
        return EachElemenAssigment(a, b, (i, j) => a.Values[i, j] - b.Values[i, j]);
    }


    private static Matrix EachElemenAssigment(Matrix a, Matrix b, Func<int, int, double> mathOperation)
    {
        if (a.RowsAmount != b.RowsAmount || a.ColumnsAmount != b.ColumnsAmount)
        {
            throw new Exception("Matrices must have the same dimensions");
        }

        Matrix result = new Matrix(a.RowsAmount, a.ColumnsAmount);

        for (int i = 0; i < a.RowsAmount; i++)
        {
            for (int j = 0; j < a.ColumnsAmount; j++)
            {
                result.Values[i, j] = mathOperation(i, j);
            }
        }

        return result;
    }
}
