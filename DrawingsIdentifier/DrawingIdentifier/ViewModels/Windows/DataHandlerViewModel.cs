using DrawingIdentifierGui.Models;
using DrawingIdentifierGui.MVVM;
using Microsoft.Win32;
using ToyNeuralNetwork;
using ToyNeuralNetwork.Math;
using ToyNeuralNetwork.QuickDrawHandler;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;

using Matrix = ToyNeuralNetwork.Math.Matrix;

namespace DrawingIdentifierGui.ViewModels.Windows;

public class DataHandlerViewModel : ViewModelBase
{
    public RelayCommand StartLoadingDataCommand => new RelayCommand(parameter =>
    {
        var openFolderDialog = new OpenFolderDialog() { Multiselect = false, Title = "Select QuickDraw data folder" };
        var dialogResult = openFolderDialog.ShowDialog();

        if (dialogResult is null || dialogResult == false)
        {
            MessageBox.Show("QuickDraw data folder not selected.", "Error", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        if (!Directory.Exists(openFolderDialog.FolderName))
        {
            MessageBox.Show("QuickDraw data folder not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        var files = Directory.GetFiles(openFolderDialog.FolderName, "*.npy")
                             .Select(f => Path.GetFileNameWithoutExtension(f))
                             .Distinct()
                             .ToArray();

        var classes = QuickDrawSet.CategoryToIndex.Keys.ToArray();
        if (files.Length != classes.Length || !classes.All(files.Contains))
        {
            MessageBox.Show("QuickDraw data folder does not contain all classes.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        MainWindowViewModel.Instance!.NotifyOnLongProcessBegin();

        cancellationTokenSource = new CancellationTokenSource();

        IsLoadingData = true;
        IsNotLoadingData = false;

        ProgressBarText = $"Loading data: 0/{App.CLASSES_AMOUNT}";

        Task.Factory.StartNew(() =>
        {
            //TODO check if randomlyShiftData is not better when false....
            var quickDrawData = ToyNeuralNetwork.QuickDrawHandler.QuickDrawDataReader.LoadQuickDrawSamplesFromDirectory(openFolderDialog.FolderName, SamplesPerFile, true, true, 255, cancellationTokenSource.Token, (i) =>
            {
                ForceMainThread(() =>
                {
                    LoadedFilesAmount = i;
                });
            });

            if (quickDrawData == null)
            {
                ForceMainThread(() =>
                {
                    LoadedFilesAmount = 0;
                });
            }
            else
            {
                ProgressBarText = "Parsing data ...";

                (var trainData, var testData) = quickDrawData.SplitIntoTrainTest();

                var trainFlat = trainData.Select(i => (new Matrix[] { MatrixExtender.FlattenMatrix(i.inputs) }, i.outputs));
                var testFlat = testData.Select(i => (new Matrix[] { MatrixExtender.FlattenMatrix(i.inputs) }, i.outputs));

                App.TestData = testData;
                App.TrainData = trainData;
                App.TrainDataFlat = trainFlat.ToArray();
                App.TestDataFlat = testFlat.ToArray();

                ForceMainThread(() =>
                {
                    ClassDrawingImagesModels = GetImages();
                });

                ProgressBarText = "Data loaded";
            }

            cancellationTokenSource = null;

            IsLoadingData = false;
            IsNotLoadingData = true;

            MainWindowViewModel.Instance!.NotifyOnLongProcessEnd();
        }, TaskCreationOptions.LongRunning);
    });

    public RelayCommand StopLoadingDataCommand => new RelayCommand(parameter =>
    {
        if (cancellationTokenSource != null)
        {
            cancellationTokenSource.Cancel();
            ProgressBarText = "Stopping...";
        }
    });

    private ObservableCollection<ClassDrawingImagesModel> classDrawingImagesModels = new ObservableCollection<ClassDrawingImagesModel>();
    public ObservableCollection<ClassDrawingImagesModel> ClassDrawingImagesModels
    {
        get => classDrawingImagesModels;
        set
        {
            classDrawingImagesModels = value;
            OnPropertyChanged();
        }
    }

    private bool isLoadingData = false;
    public bool IsLoadingData
    {
        get => isLoadingData;
        set
        {
            isLoadingData = value;
            OnPropertyChanged();
        }
    }

    private bool isNotLoadingData = true;
    public bool IsNotLoadingData
    {
        get => isNotLoadingData;
        set
        {
            isNotLoadingData = value;
            OnPropertyChanged();
        }
    }

    private CancellationTokenSource? cancellationTokenSource;

    private int samplesPerFile = 5000;
    public int SamplesPerFile
    {
        get => samplesPerFile;
        set
        {
            samplesPerFile = (Math.Max(1000, value));
            OnPropertyChanged();
        }
    }

    private int loadedFilesAmount = App.TestData.Length == 0 ? 0 : App.CLASSES_AMOUNT;
    public int LoadedFilesAmount
    {
        get => loadedFilesAmount;
        set
        {
            loadedFilesAmount = value;

            if (value == 0)
            {
                ProgressBarText = "No data loaded";
            }
            else
                ProgressBarText = $"Loading data: {value}/{App.CLASSES_AMOUNT}";

            OnPropertyChanged();
        }
    }

    private string progressBarText = App.TestData.Length == 0 ? "No data loaded" : "Data loaded";
    public string ProgressBarText
    {
        get => progressBarText;
        set
        {
            progressBarText = value;
            OnPropertyChanged();
        }
    }

    public DataHandlerViewModel()
    {
        classDrawingImagesModels = GetImages();
    }

    private ObservableCollection<ClassDrawingImagesModel> GetImages()
    {
        var res = new ObservableCollection<ClassDrawingImagesModel>();
        int classesAmount = App.CLASSES_AMOUNT;
        int samplesAmount = 10;

        if (App.TrainData.Length == 0)
        {
            return res;
        }

        for (int i = 0; i < samplesAmount; i++)
        {
            List<Matrix> row = new(classesAmount);

            for (int j = 0; j < classesAmount; j++)
            {
                var tmp = App.TrainData.Where(d => d.outputs[j, 0] == 1).Skip(i).Take(1).Select(d => d.inputs[0]);
                row.Add(tmp.First());
            }

            var classDrawingImagesModel = new ClassDrawingImagesModel(row.ToArray());
            res.Add(classDrawingImagesModel);
        }

        return res;
    }

    private void ForceMainThread(Action action)
    {
        Application.Current.Dispatcher.BeginInvoke(action);
    }
}