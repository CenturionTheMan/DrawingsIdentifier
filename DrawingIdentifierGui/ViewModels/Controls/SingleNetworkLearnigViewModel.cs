using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.ViewModels.Windows;
using Microsoft.Win32;
using NeuralNetworkLibrary;
using NeuralNetworkLibrary.QuickDrawHandler;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Xml.Linq;

namespace DrawingIdentifierGui.ViewModels.Controls
{
    public class SingleNetworkLearnigViewModel : ViewModelBase
    {
        public RelayCommand StartLearningCommand => new RelayCommand(dd =>
        {
            InitializeLearning();
        });

        public RelayCommand StopTrainingCommand => new RelayCommand(dd =>
        {
            if (TrainingCts is not null)
            {
                TrainingCts.Cancel();
                Info = "Stopping ...";
            }
        });

        public RelayCommand SaveNeuralNetwork => new RelayCommand(dd =>
        {
            var dialog = new SaveFileDialog()
            {
                AddExtension = true,
                Filter = "Xml files (*.xml)|*.xml",
                DefaultExt = ".xml",
                //InitialDirectory = Directory.GetCurrentDirectory(),
            };

            var res = dialog.ShowDialog();

            if (res is not null && res.Value == true)
            {
                float? cor = (neuralNetwork is null || learningConfig is null) ? null : neuralNetwork!.CalculateCorrectness(App.TestData.Take(1000).ToArray());
                neuralNetwork!.SaveToXmlFile(dialog.FileName, testCorrectness: cor);
                MessageBox.Show("Neural Network saved.");
            }
        });

        public RelayCommand LoadNeuralNetwork => new RelayCommand(dd =>
        {
            var dialog = new OpenFileDialog()
            {
                AddExtension = true,
                CheckFileExists = true,
                Filter = "Xml files (*.xml)|*.xml",
                DefaultExt = ".xml",
                //InitialDirectory = Directory.GetCurrentDirectory(),
            };

            if (dialog.ShowDialog() == true)
            {
                var nn = NeuralNetwork.LoadFromXmlFile(dialog.FileName);
                if (nn is null) return;

                try
                {
                    App.NeuralNetworkConfigModels[typeOfNN].LoadDataFromFile(dialog.FileName);
                    App.NeuralNetworks[typeOfNN] = nn;

                    string testCorStr = learningConfig!.TestCorrectness is null ? "unknown" : $"{learningConfig!.TestCorrectness.Value.ToString("0.00")}%";
                    string trainCorStr = learningConfig!.TrainCorrectness is null ? "unknown" : $"{learningConfig!.TrainCorrectness.Value.ToString("0.00")}%";
                    Info = $"Test correctness:  {testCorStr}{Environment.NewLine}Train correctness: {trainCorStr}";

                    MessageBox.Show("Neural Network loaded.");
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
        });

        private NeuralNetworkConfigModel? learningConfig;

        private string titleName = "Unknown";
        public string TitleName
        { get { return titleName; } set { titleName = value; OnPropertyChanged(); } }

        private int typeOfNN = -2;
        public int TypeOfNN
        { get { return typeOfNN; } set { typeOfNN = value; OnPropertyChanged(); } }

        private double epochPercentFinish = 0.0;
        public double EpochPercentFinish
        {
            get { return epochPercentFinish; }
            set { epochPercentFinish = value; OnPropertyChanged(); }
        }

        private int finishedEpochs = 0;
        public int FinishedEpochs
        {
            get { return finishedEpochs; }
            set { finishedEpochs = value; OnPropertyChanged(); }
        }

        private string finishedEpochText = $"0/?";
        public string FinishedEpochText
        { get { return finishedEpochText; } set { finishedEpochText = value; OnPropertyChanged(); } }

        private string batchError = "";
        public string BatchError
        { get => batchError; set { batchError = value; OnPropertyChanged(); } }

        private string info = $"";
        public string Info
        {
            get { return info; }
            set { info = value; OnPropertyChanged(); }
        }

        private CancellationTokenSource? trainingCts;
        public CancellationTokenSource? TrainingCts
        {
            get => trainingCts;
            set
            {
                trainingCts = value;
                IsTrainingInProgress = trainingCts != null;
                IsTrainingNotInProgress = trainingCts == null;
            }
        }

        private bool isTrainingInProgress;
        public bool IsTrainingInProgress
        {
            get => isTrainingInProgress;
            set
            {
                isTrainingInProgress = value;
                OnPropertyChanged();
            }
        }

        private bool isTrainingNotInProgress = true;
        public bool IsTrainingNotInProgress
        {
            get => isTrainingNotInProgress;
            set
            {
                isTrainingNotInProgress = value;
                OnPropertyChanged();
            }
        }

        public int EpochAmount { get => learningConfig!.EpochAmount; }

        private NeuralNetwork? neuralNetwork;

        public SingleNetworkLearnigViewModel(string name)
        {
            TitleName = "";
            var split = name.Split("_");
            for (int i = 0; i < split.Length - 1; i++)
            {
                TitleName += split[i] + " ";
            }

            TypeOfNN = int.Parse(split[^1]);

            learningConfig = App.NeuralNetworkConfigModels[TypeOfNN];
            neuralNetwork = App.NeuralNetworks[TypeOfNN];

            neuralNetwork.OnBatchTrainingIteration = (epoch, epochPercentFinish, batchError) =>
            {
                FinishedEpochs = epoch;
                EpochPercentFinish = epochPercentFinish;
                BatchError = $"Batch error: {batchError.ToString("0.0000")}";
                FinishedEpochText = $"{FinishedEpochs} / {learningConfig!.EpochAmount}";
            };

            string testCorStr = learningConfig!.TestCorrectness is null ? "unknown" : $"{learningConfig!.TestCorrectness.Value.ToString("0.00")}%";
            string trainCorStr = learningConfig!.TrainCorrectness is null ? "unknown" : $"{learningConfig!.TrainCorrectness.Value.ToString("0.00")}%";
            Info = $"Test correctness:  {testCorStr}{Environment.NewLine}Train correctness: {trainCorStr}";

            FinishedEpochText = $"{FinishedEpochs} / {learningConfig!.EpochAmount}";
        }

        private void RunLearning()
        {
            MainWindowViewModel.Instance!.NotifyOnLongProcessBegin();

            TrainingCts = new CancellationTokenSource();

            Task.Factory.StartNew(() =>
            {
                neuralNetwork!.OnTrainingFinished += () =>
                {
                    Debug.WriteLine("Finished learning");

                    Debug.WriteLine("Testing...");
                    ForceMainThread(() =>
                    {
                        Info = "Calculating correctness...";
                    });

                    var testData = neuralNetwork.IsConvolutional() ? App.TestData : App.TestDataFlat;
                    var trainData = neuralNetwork.IsConvolutional() ? App.TrainData : App.TrainDataFlat;

                    var testCorrectness = neuralNetwork.CalculateCorrectness(testData);
                    var trainCorrectness = neuralNetwork.CalculateCorrectness(trainData.Take(1000).ToArray());

                    ForceMainThread(() =>
                    {
                        Info = $"Test correctness:  {testCorrectness}%{Environment.NewLine}Train correctness: {trainCorrectness}%";
                    });

                    learningConfig!.TrainCorrectness = trainCorrectness;
                    learningConfig!.TestCorrectness = testCorrectness;

                    TrainingCts = null;
                    MainWindowViewModel.Instance!.NotifyOnLongProcessEnd();
                };

                var trainer = learningConfig!.CreateTrainer(neuralNetwork!);
                (var task, var ctsTrainer) = trainer.RunTrainingOnTask();
                TrainingCts = ctsTrainer;
            });
        }

        private void InitializeLearning()
        {
            if (App.TestData.Length == 0 || App.TrainData.Length == 0)
            {
                MessageBox.Show("No data loaded... Load data first.");
                return;
            }

            // run learing
            RunLearning();
        }

        private void ForceMainThread(Action action)
        {
            Application.Current.Dispatcher.Invoke(action);
        }
    }
}