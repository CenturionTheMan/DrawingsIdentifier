using NeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DrawingIdentifierGui.Models;

public class NeuralNetworkConfig
{
    public (double[] inputs, double[] outputs)[]? Data { get; set; }
    public int SamplesAmountToLoadPerFile { get; set; }
    public double LearningRate { get; set; }
    public int EpochAmount { get; set; }
    public int BatchSize { get; set; }
    public double ExpectedMaxError { get; set; }

    public ObservableCollection<NNLayerConfig>? NeuralNetworkLayers { get; set; }
}