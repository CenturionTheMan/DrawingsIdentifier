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
    public bool IsSizeEnable { get; set; } = true;

    private int size;
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

    private ActivationFunction? activationFunction;
    public ActivationFunction? ActivationFunction
    {
        get => activationFunction;
        set
        {
            activationFunction = value;
            OnPropertyChanged();
        }
    }

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