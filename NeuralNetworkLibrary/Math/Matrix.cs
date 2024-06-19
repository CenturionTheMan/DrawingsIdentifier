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

    public void InitializeXavier()
    {
        double limit = Math.Sqrt(6.0 / (RowsAmount + ColumnsAmount));

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                this[i, j] = random.NextDouble() * 2 * limit - limit;
            }
        }
    }

    public void InitializeHe()
    {
        double limit = Math.Sqrt(6.0 / ColumnsAmount);

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                this[i, j] = random.NextDouble() * 2 * limit - limit;
            }
        }
    }

    public double GetUnSquaredNorm()
    {
        double sum = 0;
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                sum += Values[i, j] * Values[i, j];
            }
        }
        return sum;
    }

    public double GetNorm()
    {
        return Math.Sqrt(GetUnSquaredNorm());
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

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                sb.AppendFormat("{0,5} ", Values[i, j].ToString("0.00"));
            }
            sb.Append("\n");
        }

        return sb.ToString();
    }

    public string ToFileString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                sb.Append(Values[i, j].ToString() + " ");
            }
            sb.Append("\n");
        }

        return sb.ToString();
    }

    public static bool TryParse(string matrixString, out Matrix matrix)
    {
        string[] rows = matrixString.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        double[,] values = new double[rows.Length, rows[0].Split(' ', StringSplitOptions.RemoveEmptyEntries).Length];

        for (int i = 0; i < rows.Length; i++)
        {
            string[] columns = rows[i].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < columns.Length; j++)
            {
                if(double.TryParse(columns[j], out double value) == false)
                {
                    matrix = new Matrix(0, 0);
                    return false;
                }
                if(values.GetLength(1) <= j || values.GetLength(0) <= i)
                {
                    matrix = new Matrix(0, 0);
                    return false;
                }
                values[i, j] = value;
            }
        }

        matrix = new Matrix(values);
        return true;
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

