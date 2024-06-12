using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkUnitTest.Tests;

public class UtilitiesTests
{
    [Fact]
    public void FlattenMatrix_ValidMatrix_ReturnsFlattenedMatrix()
    {
        // Arrange
        Matrix[] matrices = new Matrix[]
        {
            new Matrix(new double[,] { { 1, 2 }, { 3, 4 } }),
            new Matrix(new double[,] { { 5, 6 }, { 7, 8 } })
        };

        // Act
        Matrix flattenedMatrix = Utilities.FlattenMatrix(matrices);

        Matrix expectedResult = new Matrix([1, 2, 3, 4, 5, 6, 7, 8]);
        // Assert
        Assert.True(expectedResult.Equals(flattenedMatrix));
    }

    [Fact]
    public void UnflattenMatrix_ValidFlattenedMatrix_ReturnsUnflattenedMatrices()
    {
        // Arrange
        Matrix flattenedMatrix = new Matrix([1, 2, 3, 4, 5, 6, 7, 8]);
        int matrixSize = 2;

        // Act
        Matrix[] matrices = Utilities.UnflattenMatrix(flattenedMatrix, matrixSize);

        Matrix[] expectedMatrices = new Matrix[]
        {
            new Matrix(new double[,] { { 1, 2 }, { 3, 4 } }),
            new Matrix(new double[,] { { 5, 6 }, { 7, 8 } })
        };

        // Assert
        for (int i = 0; i < matrices.Length; i++)
        {
            Assert.True(expectedMatrices[i].Equals(matrices[i]));
        }
    }

    [Fact]
    public void FlattenUnflattenMatrix_ValidMatrix_ReturnsOriginalMatrix()
    {
        // Arrange
        Matrix[] matricesList = new Matrix[]
        {
            new Matrix(new double[,] {
                { 1, 2, 3 },
                { 4, 5, 6 } }),

            new Matrix(new double[,] {
                { 7,8,9},
                { 10, 11, 12 } })
        };
        Matrix flattened = new Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        Matrix testFlatten = Utilities.FlattenMatrix(matricesList);
        Matrix[] testUnFlatten = Utilities.UnflattenMatrix(flattened, 2, 3);

        Assert.True(flattened.Equals(testFlatten));
        for (int i = 0; i < matricesList.Length; i++)
        {
            Assert.True(matricesList[i].Equals(testUnFlatten[i]));
        }
    }
}