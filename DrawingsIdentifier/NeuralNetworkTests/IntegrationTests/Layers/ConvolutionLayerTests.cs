using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTests.IntegrationTests.Layers;

public class ConvolutionLayerTests
{
    [Fact]
    public void TestBackwardPass()
    {
        // Arrange
        var input = new Matrix(new float[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        Matrix[,] kernel = {
            {
                new Matrix(new float[,]
                {
                    { 1, 0 },
                    { 0, -1 }
                })
            }
        };

        Matrix[] bias = [new Matrix(new float[,]
        {
            { 1 }
        })];

        var layerOutputBeforeActivation = new Matrix(new float[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        var dAin = new Matrix(new float[,]
        {
            { 0.1f, 0.2f },
            { 0.3f, 0.4f }
        });

        var layer = new ConvolutionLayer((1,3,3), kernel, bias, 1, NeuralNetworkLibrary.Utils.ActivationFunction.ReLU);
        float learningRate = 0.01f;

        // Act
        ILayer tmp = layer;
        var dA = tmp.Backward(new Matrix[] { dAin }, new Matrix[] { input }, new Matrix[] { layerOutputBeforeActivation }, learningRate);

        // Assert
        Matrix[,] expectedKernel = {
            {
                new Matrix(new float[,]
                {
                    { 0.008f, 0.010f },
                    { 0.014f, 0.016f }
                })
            }
        };
        Assert.Equal(layer.kernels, layer.kernels);
    }
}
