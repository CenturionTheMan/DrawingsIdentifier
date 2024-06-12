using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public class Matrix
{
    private static Random random = new();
    private double[,] Values { get; set; }

    public readonly int RowsAmount;
    public readonly int ColumnsAmount;

    /// <summary>
    /// Creates a new Matrix filled with zeros of dimensions rows: singleColumnValues, columns: 1
    /// </summary>
    /// <param name="singleColumnValues">Rows amount</param>
    public Matrix(double[] singleColumnValues)
    {
        this.Values = new double[singleColumnValues.Length, 1];

        for (int i = 0; i < singleColumnValues.Length; i++)
        {
            Values[i, 0] = singleColumnValues[i];
        }

        RowsAmount = singleColumnValues.Length;
        ColumnsAmount = 1;
    }

    /// <summary>
    /// Creates a new Matrix based on the given values
    /// </summary>
    /// <param name="values">Values to be assigned to the matrix</param>
    public Matrix(double[,] values)
    {
        this.Values = values;
        RowsAmount = values.GetLength(0);
        ColumnsAmount = values.GetLength(1);
    }

    /// <summary>
    /// Creates a new Matrix filled with zeros of dimensions rows: rowsAmount, columns: columnsAmount
    /// </summary>
    /// <param name="rowsAmount">Rows amount</param>
    /// <param name="columnsAmount">Columns amount</param>
    public Matrix(int rowsAmount, int columnsAmount)
    {
        Values = new double[rowsAmount, columnsAmount];
        RowsAmount = rowsAmount;
        ColumnsAmount = columnsAmount;
    }

    /// <summary>
    /// Creates a new Matrix filled with random values of dimensions rows: rowsAmount, columns: columnsAmount
    /// </summary>
    /// <param name="rowsAmount">Rows amount</param>
    /// <param name="columnsAmount">Columns amount</param>
    /// <param name="min">Minimum value</param>
    /// <param name="max">Maximum value</param>
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

    /// <summary>
    /// Gives access to the value at the given indexes
    /// </summary>
    /// <param name="i">Row index</param>
    /// <param name="j">Column index</param>
    /// <returns>Value at the given indexes</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when given indexes are out of range</exception>
    public double this[int i, int j]
    {
        get
        {
            if (i < 0 || i >= RowsAmount || j < 0 || j >= ColumnsAmount)
            {
                throw new IndexOutOfRangeException($"Given indexes ([{i},{j}]) are out of range for Matrix of size: {RowsAmount}x{ColumnsAmount}.");
            }
            return Values[i, j];
        }
        set
        {
            if (i < 0 || i >= RowsAmount || j < 0 || j >= ColumnsAmount)
            {
                throw new IndexOutOfRangeException($"Given indexes ([{i},{j}]) are out of range for Matrix of size: {RowsAmount}x{ColumnsAmount}.");
            }
            Values[i, j] = value;
        }
    }

    public IEnumerator<double> GetEnumerator()
    {
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                yield return Values[i, j];
            }
        }
    }

    public bool Equals(Matrix matrix)
    {
        if (matrix.RowsAmount != RowsAmount || matrix.ColumnsAmount != ColumnsAmount)
        {
            return false;
        }

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                if (Values[i, j] != matrix.Values[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public Matrix Copy()
    {
        return new Matrix((double[,])this.Values.Clone());
    }

    public double[,] ToArray()
    {
        return (double[,])Values.Clone();
    }

    /// <summary>
    /// Creates new matrix which is result of dot product of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of dot product of two matrices</returns>
    public static Matrix DotProductMatrices(Matrix a, Matrix b)
    {
        if (a.ColumnsAmount != b.RowsAmount)
        {
            throw new ArgumentException("Number of columns in the first matrix must be equal to the number of rows in the second matrix");
        }

        return new Matrix(Accord.Math.Matrix.Dot(a.Values, b.Values));
    }

    /// <summary>
    /// Creates new matrix which is result of element-wise multiplication of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of element-wise multiplication of two matrices</returns>
    public static Matrix ElementWiseMultiplyMatrices(Matrix a, Matrix b)
    {
        if (CheckIfDimensionsAreEqual(a, b) == false)
        {
            throw new ArgumentException("Matrices must have the same dimensions");
        }
        return EachElementAssignment(a, (i, j) => a.Values[i, j] * b.Values[i, j]);
    }

    /// <summary>
    /// Creates new matrix which is result of element-wise addition of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of element-wise addition of two matrices</returns>
    public static Matrix ElementWiseAddMatrices(Matrix a, Matrix b)
    {
        if (CheckIfDimensionsAreEqual(a, b) == false)
        {
            throw new ArgumentException("Matrices must have the same dimensions");
        }
        return EachElementAssignment(a, (i, j) => a.Values[i, j] + b.Values[i, j]);
    }

    /// <summary>
    /// Creates new matrix which is result of element-wise subtraction of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of element-wise subtraction of two matrices</returns>
    public static Matrix ElementWiseSubtractMatrices(Matrix a, Matrix b)
    {
        if (CheckIfDimensionsAreEqual(a, b) == false)
        {
            throw new ArgumentException("Matrices must have the same dimensions");
        }
        return EachElementAssignment(a, (i, j) => a.Values[i, j] - b.Values[i, j]);
    }

    public static Matrix operator *(Matrix a, double b)
    {
        return EachElementAssignment(a, (i, j) => a.Values[i, j] * b);
    }

    public static Matrix operator +(Matrix a, double b)
    {
        return EachElementAssignment(a, (i, j) => a.Values[i, j] + b);
    }

    /// <summary>
    /// Creates new matrix which is result of applying given function to each element of the a matrix.
    /// </summary>
    /// <param name="a">Matrix to apply function to</param>
    /// <param name="b">Second matrix. Method will use if only for checking dimensions</param>
    /// <param name="mathOperation">Function to apply</param>
    private static Matrix EachElementAssignment(Matrix a, Func<int, int, double> mathOperation)
    {
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

    /// <summary>
    /// Checks if given matrices have the same dimensions
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>True if dimensions are the same, false otherwise</returns>
    private static bool CheckIfDimensionsAreEqual(Matrix a, Matrix b)
    {
        return a.RowsAmount == b.RowsAmount && a.ColumnsAmount == b.ColumnsAmount;
    }
}

public static class MatrixExtender
{
    public static Matrix ConvolutionFull(this Matrix input, Matrix kernel, int stride)
    {
        int outputRows = input.RowsAmount + kernel.RowsAmount - 1;
        int outputColumns = input.ColumnsAmount + kernel.ColumnsAmount - 1;
        Matrix output = new Matrix(outputRows, outputColumns);

        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputColumns; j++)
            {
                double sum = 0;

                for (int m = 0; m < kernel.RowsAmount; m++)
                {
                    for (int n = 0; n < kernel.ColumnsAmount; n++)
                    {
                        int x = i * stride - m;
                        int y = j * stride - n;

                        if (x >= 0 && x < input.RowsAmount && y >= 0 && y < input.ColumnsAmount)
                        {
                            sum += input[x, y] * kernel[m, n];
                        }
                    }
                }

                output[i, j] = sum;
            }
        }
        return output;
    }

    public static Matrix ConvolutionValid(this Matrix input, Matrix kernel, int stride)
    {
        var rot = kernel.Rotate180();
        return input.CrossCorrelationValid(rot, stride);
    }

    public static Matrix CrossCorrelationFull(this Matrix input, Matrix kernel, int stride)
    {
        var rot = kernel.Rotate180();
        return input.ConvolutionFull(rot, stride);
    }

    public static Matrix CrossCorrelationValid(this Matrix input, Matrix kernel, int stride)
    {
        int outputRows = (input.RowsAmount - kernel.RowsAmount) / stride + 1;
        int outputColumns = (input.ColumnsAmount - kernel.ColumnsAmount) / stride + 1;
        Matrix output = new Matrix(outputRows, outputColumns);

        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputColumns; j++)
            {
                double sum = 0;

                for (int m = 0; m < kernel.RowsAmount; m++)
                {
                    for (int n = 0; n < kernel.ColumnsAmount; n++)
                    {
                        int x = i * stride + m;
                        int y = j * stride + n;

                        if (x >= 0 && x < input.RowsAmount && y >= 0 && y < input.ColumnsAmount)
                        {
                            sum += input[x, y] * kernel[m, n];
                        }
                    }
                }

                output[i, j] = sum;
            }
        }
        return output;
    }

    public static Matrix AddPadding(this Matrix input, int padding)
    {
        Matrix output = new Matrix(input.RowsAmount + 2 * padding, input.ColumnsAmount + 2 * padding);

        for (int i = 0; i < input.RowsAmount; i++)
        {
            for (int j = 0; j < input.ColumnsAmount; j++)
            {
                output[i + padding, j + padding] = input[i, j];
            }
        }

        return output;
    }

    /// <summary>
    /// Rotates the matrix by 180 degrees
    /// </summary>
    /// <returns>New matrix after rotation</returns>
    public static Matrix Rotate180(this Matrix matrix)
    {
        Matrix result = new Matrix(matrix.RowsAmount, matrix.ColumnsAmount);
        for (int i = 0; i < matrix.RowsAmount; i++)
        {
            for (int j = 0; j < matrix.ColumnsAmount; j++)
            {
                result[i, j] = matrix[matrix.RowsAmount - i - 1, matrix.ColumnsAmount - j - 1];
            }
        }
        return result;
    }

    public static int IndexOfMax(this Matrix matrix)
    {
        if (matrix.ColumnsAmount != 1)
            throw new ArgumentException("Matrix must have only one column");

        double max = double.MinValue;
        int index = 0;
        for (int i = 0; i < matrix.RowsAmount; i++)
        {
            if (matrix[i, 0] > max)
            {
                max = matrix[i, 0];
                index = i;
            }
        }
        return index;
    }

    /// <summary>
    /// Sums all elements of the matrix
    /// </summary>
    /// <param name="a">Matrix to sum</param>
    /// <returns>Sum of all elements</returns>
    public static double Sum(this Matrix a)
    {
        double sum = 0;
        foreach (var item in a)
        {
            sum += item;
        }
        return sum;
    }

    /// <summary>
    /// Finds the maximum value in the matrix
    /// </summary>
    /// <param name="a">Matrix to search</param>
    /// <returns>Maximum value in the matrix</returns>
    public static double Max(this Matrix a)
    {
        double max = double.MinValue;
        foreach (var item in a)
        {
            if (item > max)
            {
                max = item;
            }
        }
        return max;
    }

    /// <summary>
    /// Transposes the matrix
    /// </summary>
    /// <param name="a">Matrix to transpose</param>
    /// <returns>Transposed matrix</returns>
    public static Matrix Transpose(this Matrix a)
    {
        Matrix result = new Matrix(a.ColumnsAmount, a.RowsAmount);

        for (int i = 0; i < a.ColumnsAmount; i++)
        {
            for (int j = 0; j < a.RowsAmount; j++)
            {
                result[i, j] = a[j, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Applies the given function to each element of the matrix
    /// </summary>
    /// <param name="a">Matrix to apply function to</param>
    /// <param name="function">Function to apply</param>
    /// <returns>Matrix with applied function</returns>
    public static Matrix ApplyFunction(this Matrix a, Func<double, double> function)
    {
        Matrix result = new Matrix(a.RowsAmount, a.ColumnsAmount);

        for (int i = 0; i < a.RowsAmount; i++)
        {
            for (int j = 0; j < a.ColumnsAmount; j++)
            {
                result[i, j] = function(a[i, j]);
            }
        }
        return result;
    }

    public static Matrix DotProduct(this Matrix a, Matrix b)
    {
        return Matrix.DotProductMatrices(a, b);
    }

    public static Matrix ElementWiseMultiply(this Matrix a, Matrix b)
    {
        return Matrix.ElementWiseMultiplyMatrices(a, b);
    }

    public static Matrix ElementWiseAdd(this Matrix a, Matrix b)
    {
        return Matrix.ElementWiseAddMatrices(a, b);
    }

    public static Matrix ElementWiseSubtract(this Matrix a, Matrix b)
    {
        return Matrix.ElementWiseSubtractMatrices(a, b);
    }
}