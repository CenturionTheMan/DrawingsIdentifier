using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.Views.Controls;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkLibrary;
using System.Collections.ObjectModel;

namespace DrawingIdentifierGui.ViewModels.Windows;

public class FeedForwardConfigViewModel : ViewModelBase
{
    public ObservableCollection<NNLayerConfig> NetworkLayers
    {
        get
        {
            return App.FeedForwardNNConfig.NeuralNetworkLayers!;
        }
        set
        {
            App.FeedForwardNNConfig.NeuralNetworkLayers = value;
            OnPropertyChanged();
        }
    }

    public FeedForwardConfigViewModel()
    {
    }
}