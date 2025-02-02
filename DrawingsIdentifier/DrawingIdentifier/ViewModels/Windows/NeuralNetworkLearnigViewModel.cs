﻿using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.Views.Controls;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using ToyNeuralNetwork;
using System.Windows;

namespace DrawingIdentifierGui.ViewModels.Windows;

internal class NeuralNetworkLearnigViewModel : ViewModelBase
{
    private bool isLearning;
    public bool IsLearning 
    { 
        get => isLearning; 
        set
        {
            isLearning = value;
            OnPropertyChanged();
        }
    }

    public NeuralNetworkLearnigViewModel()
    {
        
    }
}