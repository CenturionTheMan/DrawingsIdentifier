﻿using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using Microsoft.Win32;
using NeuralNetworkLibrary;
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

        private NeuralNetworkConfig? learningConfig;

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

        public int EpochAmount { get => App.FeedForwardNNConfig.EpochAmount; }

        private INeuralNetwork? neuralNetwork;

        public SingleNetworkLearnigViewModel(string name)
        {
            string tmp = name.Replace("Neural_Network", "");
            TitleName = tmp.Replace("_", " ").Trim() + Environment.NewLine + "Neural Network";
            TypeOfNN = name.Contains("Convo") ? 1 : 0;

            RefreshNN(TypeOfNN, false);

            FinishedEpochText = $"{FinishedEpochs} / {learningConfig!.EpochAmount}";
        }

        private void RefreshNN(int nnType, bool createNewInstance)
        {
            learningConfig = nnType switch
            {
                0 => App.FeedForwardNNConfig,
                1 => App.ConvolutionalNNConfig,
                _ => throw new Exception("Unknown neural network type")
            };

            switch (nnType)
            {
                case 0:
                    {
                        if (createNewInstance)
                        {
                            var ctorData = learningConfig!.GetLayersAsCtorData();
                            App.FeedForwardNN = new FeedForwardNeuralNetwork(ctorData.layersSize, ctorData.activationFunctions);
                        }
                        neuralNetwork = App.FeedForwardNN;
                        break;
                    }
                case 1:
                    {
                        neuralNetwork = null; //here assign convolutional neural network
                        break;
                    }
                default:
                    throw new NotImplementedException("Unknown nn type");
            }

            if (neuralNetwork == null)
            {
                return;
            }

            double batchErr = double.MaxValue;
            neuralNetwork!.OnLearningIteration = (epoch, epochPercentFinish, batchError) =>
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

        private void RunLearning(string[] files)
        {
            Task.Factory.StartNew(() =>
            {
                var quickDrawData = ImagesProcessor.DataReader.LoadQuickDrawSamplesFromFiles(files, learningConfig!.SamplesAmountToLoadPerFile);
                if (quickDrawData == null)
                {
                    return;
                }

                (var trainData, var testData) = quickDrawData.SplitIntoTrainTest();

                neuralNetwork!.Train(trainData, learningConfig!.LearningRate, learningConfig.EpochAmount, learningConfig.BatchSize, learningConfig.ExpectedMaxError);

                Debug.WriteLine("Finished learning");

                Debug.WriteLine("Testing...");
                int guessed = 0;
                foreach (var item in testData)
                {
                    var prediction = neuralNetwork!.Predict(item.inputs);
                    var max = prediction.Max();
                    int indexOfMaxPrediction = prediction.ToList().IndexOf(max);

                    var expectedMax = item.outputs.Max();
                    int indexOfMaxExpected = item.outputs.ToList().IndexOf(expectedMax);

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
            RefreshNN(this.TypeOfNN, true);

            // run learing
            RunLearning(files);
        }

        private void ForceMainThread(Action action)
        {
            Application.Current.Dispatcher.Invoke(action);
        }
    }
}