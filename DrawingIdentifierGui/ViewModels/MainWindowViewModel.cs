using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.ViewModels.Windows;
using NeuralNetworkLibrary;
using System.Runtime.CompilerServices;
using System.Windows;

namespace DrawingIdentifierGui.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
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
        public RelayCommand ShowNeuralNetwork1ConfigCommand => new RelayCommand(parameter => SelectedViewModel = new NeuralNetworkConfigViewModel(0));
        public RelayCommand ShowNeuralNetwork2ConfigCommand => new RelayCommand(parameter => SelectedViewModel = new NeuralNetworkConfigViewModel(1));
        public RelayCommand ShowDataHandlerCommand => new RelayCommand(parameter => SelectedViewModel = new DataHandlerViewModel());

        public RelayCommand ExitCommand => new RelayCommand(parameter => MainWindow.Instance.Close());
        public RelayCommand MinimalizeCommand => new RelayCommand(parameter => MainWindow.Instance.WindowState = System.Windows.WindowState.Minimized);

        private int trainingsAmount = 0;
        private bool isNotTraining = true;
        public bool IsNotTraining
        {
            get { return isNotTraining; }
            set { isNotTraining = value; OnPropertyChanged(); }
        }

        internal void NotifyOnLongProcessEnd()
        {
            trainingsAmount--;
            if (trainingsAmount == 0)
            {
                IsNotTraining = true;
            }
        }

        internal void NotifyOnLongProcessBegin()
        {
            IsNotTraining = false;
            trainingsAmount++;
        }

        public static MainWindowViewModel? Instance;

        public MainWindowViewModel()
        {
            MainWindowViewModel.Instance = this;

           
        }
    }
}