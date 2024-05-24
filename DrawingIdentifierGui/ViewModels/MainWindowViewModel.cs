using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.ViewModels.Windows;
using NeuralNetworkLibrary;

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
        public RelayCommand ShowFeedForwardNNConfigCommand => new RelayCommand(parameter => SelectedViewModel = new FeedForwardConfigViewModel());

        public RelayCommand ExitCommand => new RelayCommand(parameter => MainWindow.Instance.Close());
        public RelayCommand MinimalizeCommand => new RelayCommand(parameter => MainWindow.Instance.WindowState = System.Windows.WindowState.Minimized);
    }
}