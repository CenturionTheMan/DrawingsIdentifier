using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.ViewModels.Controls;
using DrawingIdentifierGui.Views.Controls;
using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace DrawingIdentifierGui.ViewModels
{
    internal class MainWindowViewModel : ViewModelBase
    {
        private ViewModelBase selectedViewModel = new PredictionsCanvasViewModel();

        public ViewModelBase SelectedViewModel
        {
            get { return selectedViewModel; }
            set 
            { 
                selectedViewModel = value; 
                OnPropertyChanged();
            }
        }

        public RelayCommand ShowPredictionCanvasCommnad => new RelayCommand(parameter => SelectedViewModel = new PredictionsCanvasViewModel());
        public RelayCommand ShowNeuralNetworkLearningCommand => new RelayCommand(parameter => SelectedViewModel = new NeuralNetworkLearnigViewModel());

        public RelayCommand ExitCommand => new RelayCommand(parameter => MainWindow.Instance.Close());
        public RelayCommand MinimalizeCommand => new RelayCommand(parameter => MainWindow.Instance.WindowState = System.Windows.WindowState.Minimized);
        

    }
}
