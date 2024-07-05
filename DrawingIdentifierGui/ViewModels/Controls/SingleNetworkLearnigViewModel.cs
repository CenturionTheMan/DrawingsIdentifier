using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using Microsoft.Win32;
using NeuralNetworkLibrary;
using NeuralNetworkLibrary.QuickDrawHandler;
using System.Diagnostics;
using System.Windows;

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
            }
        });

        public RelayCommand SaveNeuralNetwork => new RelayCommand(dd =>
        {
            var dialog = new SaveFileDialog() {
                AddExtension = true,
                CheckFileExists = true,
                Filter = "Xml files (*.xml)|*.xml",
                DefaultExt = ".xml",
                InitialDirectory = "./",
            };

            if(dialog.ShowDialog() == true)
            {
                neuralNetwork!.SaveToXmlFile(dialog.FileName);
            }
        });

        public RelayCommand LoadNeuralNetwork => new RelayCommand(dd => {
            //TODO
            //load nn to nn object
            //load nn to gui config
            throw new NotImplementedException();
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
            set { 
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

            RefreshNN(TypeOfNN);

            FinishedEpochText = $"{FinishedEpochs} / {learningConfig!.EpochAmount}";
        }

        private void RefreshNN(int nnType)
        {
            learningConfig = App.NeuralNetworkConfigModels[nnType];
            neuralNetwork = App.NeuralNetworks[nnType];

            neuralNetwork.OnBatchTrainingIteration = (epoch, epochPercentFinish, batchError) =>
            {
                FinishedEpochs = epoch;
                EpochPercentFinish = epochPercentFinish;
                BatchError = $"Batch error: {batchError.ToString("0.0000")}";
                FinishedEpochText = $"{FinishedEpochs} / {learningConfig!.EpochAmount}";
            };
        }

        private string[]? GetFilesForLearning()
        {
            //TODO first try to find files in some defalut directory ...

            var folderDialog = new OpenFileDialog() { Filter = "Numpy files (*.npy)|*.npy", DefaultExt = ".npy", Multiselect = true };

            bool? isFile = folderDialog.ShowDialog();
            if (isFile is null || isFile == false)
            {
                return null;
            }
            var files = folderDialog.FileNames;
            return files;
        }

        private void RunLearning(string[] files)
        {
            TrainingCts = new CancellationTokenSource();

            Task.Factory.StartNew(() =>
            {
                ForceMainThread(() =>
                {
                    Info = $"Loading files 0 / {files.Length} ...";
                });

                var quickDrawData = NeuralNetworkLibrary.QuickDrawHandler.QuickDrawDataReader.LoadQuickDrawSamplesFromFiles(files, learningConfig.SamplesPerFile, true, true, 255, TrainingCts.Token, (i) =>
                {
                    ForceMainThread(() =>
                    {
                        Info = $"Loading files {i} / {files.Length} ...";
                    });
                });

                if (quickDrawData == null)
                {
                    TrainingCts = null;
                    return;
                }

                (var trainData, var testData) = quickDrawData.SplitIntoTrainTest();
                learningConfig.TrainData = trainData;
                learningConfig.TestData = testData;


                ForceMainThread(() =>
                {
                    Info = $"";
                });

                neuralNetwork!.OnTrainingFinished += () =>
                {
                    Debug.WriteLine("Finished learning");

                    Debug.WriteLine("Testing...");
                    ForceMainThread(() =>
                    {
                        Info = $"Achieved predictions correctness: {neuralNetwork.CalculateCorrectness(learningConfig.TestData).ToString("0.00")}%";
                    });

                    TrainingCts = null;
                };

                var trainer = learningConfig.CreateTrainer(neuralNetwork!);
                (var task, var ctsTrainer) = trainer.RunTrainingOnTask();
                TrainingCts = ctsTrainer;
            });
        }

        private void InitializeLearning()
        {
            // get files
            var files = GetFilesForLearning();
            if (files == null)
            {
                return;
            }

            // refresh neural network config
            RefreshNN(this.TypeOfNN);

            // run learing
            RunLearning(files);
        }

        private void ForceMainThread(Action action)
        {
            Application.Current.Dispatcher.Invoke(action);
        }
    }
}