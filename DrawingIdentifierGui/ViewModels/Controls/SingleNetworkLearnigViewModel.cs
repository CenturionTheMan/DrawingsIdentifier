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
            Task.Factory.StartNew(InitializeLearning);
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

        private string batchError = "Min Batch error: ???";
        public string BatchError
        { get => batchError; set { batchError = value; OnPropertyChanged(); } }

        private string corectness = $"Achieved correctness: ???.??%";
        public string Correctness
        {
            get { return corectness; }
            set { corectness = value; OnPropertyChanged(); }
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

            double batchErr = double.MaxValue;
            neuralNetwork!.OnBatchTrainingIteration = (epoch, epochPercentFinish, batchError) =>
            {
                FinishedEpochs = epoch;
                EpochPercentFinish = epochPercentFinish;
                if (batchError < batchErr)
                {
                    batchErr = batchError;
                    BatchError = $"Min Batch error: {batchError.ToString("0.00")}";
                }
                FinishedEpochText = $"{FinishedEpochs} / {learningConfig!.EpochAmount}";
            };
        }

        private string[]? GetFilesForLearning()
        {
            var folderDialog = new OpenFileDialog() { Filter = "Numpy files (*.npy)|*.npy", DefaultExt = ".npy", Multiselect = true };

            bool? isFile = folderDialog.ShowDialog();
            if (isFile is null || isFile == false)
            {
                return null;
            }
            var files = folderDialog.FileNames;
            return files;
        }

        //TODO manage files / data loading -> variable for storing i model
        private void RunLearning(string[] files)
        {
            Task.Factory.StartNew(() =>
            {
                var quickDrawData = NeuralNetworkLibrary.QuickDrawHandler.QuickDrawDataReader.LoadQuickDrawSamplesFromFiles(files, learningConfig!.SamplesPerFile);
                if (quickDrawData == null)
                {
                    return;
                }

                (var trainData, var testData) = quickDrawData.SplitIntoTrainTest();

                var nn = App.NeuralNetworks[TypeOfNN];
                var trainer = App.NeuralNetworkConfigModels[TypeOfNN].CreateTrainer(nn);
                (var task, var cts) = trainer.RunTrainingOnTask();
                task.Wait();

                Debug.WriteLine("Finished learning");

                Debug.WriteLine("Testing...");
                int guessed = 0;
                foreach (var item in testData)
                {
                    var prediction = neuralNetwork!.Predict(item.inputs);
                    var max = prediction.Max();
                    int indexOfMaxPrediction = prediction.IndexOfMax();

                    var expectedMax = item.outputs.Max();
                    int indexOfMaxExpected = item.outputs.IndexOfMax();

                    if (indexOfMaxPrediction == indexOfMaxExpected)
                    {
                        guessed++;
                    }

                    ForceMainThread(() =>
                    {
                        Correctness = $"Achieved predictions correctness: {((double)guessed * 100 / testData.Count()).ToString("0.00")}%";
                    });
                }

                ForceMainThread(() =>
                {
                    Correctness = $"Achieved predictions correctness: {((double)guessed * 100 / testData.Count()).ToString("0.00")}%";
                });

                Debug.WriteLine($"Achieved predictions correctness: {((double)guessed * 100 / testData.Count()).ToString("0.00")}%");
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