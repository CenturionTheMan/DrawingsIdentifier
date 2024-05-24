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

    public ActivationFunction? ActivationFunction { get; set; }
    public string LayerName { get; set; } = "Hidden Layer";
}