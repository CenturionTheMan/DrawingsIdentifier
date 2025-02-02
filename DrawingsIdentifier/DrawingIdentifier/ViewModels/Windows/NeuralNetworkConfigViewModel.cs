﻿using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace DrawingIdentifierGui.ViewModels.Windows
{
    public class NeuralNetworkConfigViewModel : ViewModelBase
    {
        public RelayCommand AddLayerCommand => new RelayCommand((obj) =>
        {
            LayerModel layer = new();
            NeuralNetworkLayers.Add(layer);
        });

        public RelayCommand RemoveLayerCommand => new RelayCommand((obj) =>
        {
            if (SelectedLayer == null || NeuralNetworkLayers.Count <= 1) return;

            NeuralNetworkLayers.Remove(SelectedLayer);
        });

        public RelayCommand ChooseDirectoryForLogsCommand => new RelayCommand((obj) =>
        {
            var folderDialog = new OpenFolderDialog
            {
                Multiselect = false,
            };

            if (folderDialog.ShowDialog() == true)
            {
                var folderName = folderDialog.FolderName;
                SaveDirectoryPath = folderName + "/";
            }
        });

        public RelayCommand SaveChangesToNN => new RelayCommand((obj) =>
        {
            try
            {
                var nn = NeuralNetworkConfigModel.CreateNeuralNetwork(NeuralNetworkLayers.ToArray());

                App.NeuralNetworkConfigModels[type].NeuralNetworkLayers = NeuralNetworkLayers;
                App.NeuralNetworks[type] = nn;

                MessageBox.Show("Neural Network config saved. Learning progress lost.");
            }
            catch (Exception ex)
            {
                //TODO show info that nn can not be created

                NeuralNetworkLayers = App.NeuralNetworkConfigModels[type]!.NeuralNetworkLayers!;

                MessageBox.Show(ex.Message);
            }
        });

        private int type;
        public string TypeString
        {
            get
            {
                return type switch
                {
                    0 => "Neural Network I",
                    1 => "Neural Network II",
                    _ => "Unknown"
                };
            }
        }

        private LayerModel selectedLayer;
        public LayerModel SelectedLayer
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

        public bool IsSavingToLog
        {
            get
            {
                return App.NeuralNetworkConfigModels[type].SaveToLog;
            }
            set
            {
                App.NeuralNetworkConfigModels[type].SaveToLog = value;
                OnPropertyChanged();
            }
        }

        public string SaveDirectoryPath
        {
            get
            {
                return App.NeuralNetworkConfigModels[type].SaveDirectoryPath;
            }
            set
            {
                App.NeuralNetworkConfigModels[type].SaveDirectoryPath = value;
                OnPropertyChanged();
            }
        }

        public bool IsPatience
        {
            get
            {
                return App.NeuralNetworkConfigModels[type].IsPatience;
            }
            set
            {
                App.NeuralNetworkConfigModels[type].IsPatience = value;
                OnPropertyChanged();
            }
        }

        public NeuralNetworkConfigModel NeuralNetworkConfig
        {
            get
            {
                return App.NeuralNetworkConfigModels[type];
            }
            set
            {
                App.NeuralNetworkConfigModels[type] = value;
                OnPropertyChanged();
            }
        }

        private ObservableCollection<LayerModel> neuralNetworkLayers { get; set; }
        public ObservableCollection<LayerModel> NeuralNetworkLayers
        {
            get
            {
                return neuralNetworkLayers;
            }
            set
            {
                neuralNetworkLayers = value;
                OnPropertyChanged();
            }
        }

        public NeuralNetworkConfigViewModel(int type)
        {
            this.type = type;
            neuralNetworkLayers = new();
            foreach (var item in App.NeuralNetworkConfigModels[type]!.NeuralNetworkLayers!)
            {
                var copy = new LayerModel()
                {
                    ActivationFunction = item.ActivationFunction,
                    DropoutRate = item.DropoutRate,
                    IsActivationFunctionVisable = item.IsActivationFunctionVisable,
                    IsDropoutRateVisable = item.IsDropoutRateVisable,
                    IsLayerSizeVisable = item.IsLayerSizeVisable,
                    IsPoolStrideVisable = item.IsPoolStrideVisable,
                    KernelDepth = item.KernelDepth,
                    KernelSize = item.KernelSize,
                    LayerSize = item.LayerSize,
                    LayerType = item.LayerType,
                    PoolSize = item.PoolSize,
                    PoolStride = item.PoolStride
                };
                NeuralNetworkLayers.Add(copy);
            }

            selectedLayer = NeuralNetworkLayers.First();
        }
    }
}