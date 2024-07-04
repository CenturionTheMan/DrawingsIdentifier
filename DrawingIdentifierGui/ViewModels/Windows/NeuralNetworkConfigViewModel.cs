using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DrawingIdentifierGui.ViewModels.Windows
{
    public class NeuralNetworkConfigViewModel : ViewModelBase
    {
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

        private int type;

        public RelayCommand AddLayerCommand => new RelayCommand((obj) =>
        {
            LayerModel layer = new();
            App.NeuralNetworkConfigModels[type].NeuralNetworkLayers?.Insert(App.NeuralNetworkConfigModels[type].NeuralNetworkLayers.Count() - 1, layer);
        });

        public RelayCommand RemoveLayerCommand => new RelayCommand((obj) =>
        {
            if (SelectedLayer == null || App.NeuralNetworkConfigModels[type].NeuralNetworkLayers!.Count <= 1) return;

            App.NeuralNetworkConfigModels[type].NeuralNetworkLayers!.Remove(SelectedLayer);
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
                SaveDirectoryPath = folderName;
            }
        });

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

        public NeuralNetworkConfigViewModel(int type)
        {
            this.type = type;
        }
    }
}