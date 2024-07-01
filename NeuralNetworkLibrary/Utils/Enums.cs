using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary;

public enum ActivationFunction
{
    ReLU,
    Sigmoid,
    Softmax,
}

internal enum LayerType
{
    Convolution,
    Pooling,
    FullyConnected,
    Dropout,
    Reshape
}