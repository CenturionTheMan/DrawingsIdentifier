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
using System.Windows.Input;

namespace DrawingIdentifierGui.ViewModels.Windows;

public class FeedForwardConfigViewModel : ViewModelBase
{
    public RelayCommand AddLayerCommand => new RelayCommand((obj) =>
    {
        NNLayerConfig layer = new NNLayerConfig();
        App.FeedForwardNNConfig.NeuralNetworkLayers!.Insert(App.FeedForwardNNConfig.NeuralNetworkLayers.Count()-1, layer);
    });

    public RelayCommand RemoveLayerCommand => new RelayCommand((obj) =>
    {
        if(SelectedLayer == null || SelectedLayer.IsSizeEnable == false || App.FeedForwardNNConfig.NeuralNetworkLayers!.Count <=3) return;

        App.FeedForwardNNConfig.NeuralNetworkLayers!.Remove(SelectedLayer);
    });


    public NNLayerConfig SelectedLayer
    {
        get
        {
            return selectedLayer;
        }
        set
        {
            selectedLayer = value;
            OnPropertyChanged();
        }
    }
    private NNLayerConfig selectedLayer;

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