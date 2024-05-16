using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.Views.Controls;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using NeuralNetworkLibrary;
using MyBaseLibrary;
using System.Windows;

namespace DrawingIdentifierGui.ViewModels.Windows;

class NeuralNetworkLearnigViewModel : ViewModelBase
{

    public RelayCommand StartLearningCommand => new RelayCommand(dd =>
    {

        Task.Factory.StartNew(InitializeLearning);
    });


    private string status = "None";
    public string Status
    {
        get { return status; }
        set { status = value; OnPropertyChanged(); }
    }


    private string epochNumber = "Epoch number: ??";
    public string EpochNumber
    {
        get { return epochNumber; }
        set { epochNumber = value; OnPropertyChanged(); }
    }

    private string epochPercentFinished = "Epoch percent finished: ??";
    public string EpochPercentFinished
    {
        get { return epochPercentFinished; }
        set { epochPercentFinished = value; OnPropertyChanged(); }
    }

    private string batchError = "Batch error: ??";
    public string BatchError
    {
        get { return batchError; }
        set { batchError = value; OnPropertyChanged(); }
    }





    private void InitializeLearning()
    {
        const string MnistDataDirPath = "D:\\GoogleDriveMirror\\Projects\\NeuralNetworkProject\\mnist_data\\";


        Status = "Loading data...";

        var trainData = GetMnistData(MnistDataDirPath + "mnist_train_data1.csv", MnistDataDirPath + "mnist_train_data2.csv");
        var testData = GetMnistData(MnistDataDirPath + "mnist_test_data.csv");

        Status = "Neural network set up";

        Status = "Training...";
        App.NeuralNetwork.Train(trainData, 0.01, 30, 50, 0.01);

        Status = "Testing...";
        int guessed = 0;
        foreach (var item in testData)
        {
            var prediction = App.NeuralNetwork.Predict(item.inputs);
            var max = prediction.Max();
            int indexOfMaxPrediction = prediction.ToList().IndexOf(max);

            var expectedMax = item.outputs.Max();
            int indexOfMaxExpected = item.outputs.ToList().IndexOf(expectedMax);

            if (indexOfMaxPrediction == indexOfMaxExpected)
            {
                guessed++;
            }
        }

        Status = $"Correctness: {(guessed * 100.0 / testData.Length).ToString("0.00")}%";
    }


    private static (double[] inputs, double[] outputs)[] GetMnistData(params string[] paths)
    {
        var result = new List<(double[] inputs, double[] outputs)>();

        foreach (var path in paths)
        {
            var data = FilesCreatorHelper.ReadInputFromCSV(path, ',').Skip(1);

            foreach (var item in data)
            {
                double[] inputs = new double[784];
                double[] expected = new double[10];

                int numer = int.Parse(item[0]);
                expected[numer] = 1;

                inputs = item.Skip(1).Select(x => double.Parse(x) / 255.0).ToArray();

                result.Add((inputs, expected));
            }
        }

        return result.ToArray();
    }


    public NeuralNetworkLearnigViewModel()
    {
        App.NeuralNetwork.OnLearningIteration = (epoch, epochPercentFinish, batchError) =>
        {
            EpochNumber = $"Epoch number: {epoch}";
            EpochPercentFinished = $"Epoch percent finished: {epochPercentFinish.ToString("0.00")}%";
            BatchError = $"Batch error: {batchError.ToString("0.00")}";
        };
    }


    private void ForceMainThread(Action action)
    {
        Application.Current.Dispatcher.Invoke(action);
    }


}
