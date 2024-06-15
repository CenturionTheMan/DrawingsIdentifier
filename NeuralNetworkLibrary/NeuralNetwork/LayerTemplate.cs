using System.Data;

namespace NeuralNetworkLibrary;

public class LayerTemplate
{   
    internal LayerType LayerType => layerType;
    internal ActivationFunction ActivationFunction => activationFunction;

    internal int LayerSize => layerSize;

    internal int Stride => stride;
    internal int PoolSize => poolSize;
    internal int KernelSize => kernelSize;
    internal int Depth => depth;


    private LayerType layerType;
    private ActivationFunction activationFunction;

    private int layerSize;

    private int stride;
    private int poolSize;
    private int kernelSize;
    private int depth;


    private LayerTemplate()
    {

    }

    public static LayerTemplate CreateFullyConnectedLayer(int layerSize, ActivationFunction activationFunction)
    {
        return new LayerTemplate
        {
            layerType = LayerType.FullyConnected,
            activationFunction = activationFunction,
            layerSize = layerSize
        };
    }

    public static LayerTemplate CreatePoolingLayer(int poolSize, int stride)
    {
        return new LayerTemplate
        {
            layerType = LayerType.Pooling,
            poolSize = poolSize,
            stride = stride
        };
    }

    public static LayerTemplate CreateConvolutionLayer(int kernelSize, int depth, int stride, ActivationFunction activationFunction)
    {
        return new LayerTemplate
        {
            layerType = LayerType.Convolution,
            kernelSize = kernelSize,
            depth = depth,
            stride = stride,
            activationFunction = activationFunction,
        };
    }
}