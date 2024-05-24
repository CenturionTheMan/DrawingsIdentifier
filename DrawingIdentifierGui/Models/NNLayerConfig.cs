using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DrawingIdentifierGui.MVVM;
using NeuralNetworkLibrary;

namespace DrawingIdentifierGui.Models;

public class NNLayerConfig : ViewModelBase
{
    public int Size
    {
        get
        {
            return size;
        }
        set
        {
            size = value;
            OnPropertyChanged();
        }
    }

    private int size;

    public bool IsSizeEnable { get; set; } = true;

    public ActivationFunction? ActivationFunction
    {
        get => activationFunction;
        set
        {
            activationFunction = value;
            OnPropertyChanged();
        }
    }

    private ActivationFunction? activationFunction;

    public bool IsActivationFunctionEnable { get; set; } = true;

    public string LayerName { get; set; } = "Hidden Layer";

    public IEnumerable<ActivationFunction> ActivationFunctions
    {
        get
        {
            yield return NeuralNetworkLibrary.ActivationFunction.ReLU;
            yield return NeuralNetworkLibrary.ActivationFunction.Sigmoid;

            if (!IsActivationFunctionEnable)
                yield return NeuralNetworkLibrary.ActivationFunction.Softmax;
        }
    }
}