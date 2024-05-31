using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public class ConvolutionalHiddenLayers
{
    private (ConvolutionLayer ConvolutionLayer, PoolingLayer poolingLayer)[] layers;

    public ConvolutionalHiddenLayers(params (ConvolutionLayer ConvolutionLayer, PoolingLayer poolingLayer)[] layers)
    {
        this.layers = layers;
    }
}

public class ConvolutionalClassificationLayers
{
}