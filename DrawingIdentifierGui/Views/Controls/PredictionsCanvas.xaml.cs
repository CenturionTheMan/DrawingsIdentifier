using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using DrawingIdentifierGui.Utilities;
using DrawingIdentifierGui.ViewModels.Controls;
using static DrawingIdentifierGui.Utilities.BitmapCustomExtender;

namespace DrawingIdentifierGui.Views.Controls;

/// <summary>
/// Interaction logic for PredictionsCanvas.xaml
/// </summary>
public partial class PredictionsCanvas : UserControl
{
    public static PredictionsCanvas Instance;

    public PredictionsCanvas()
    {
        Instance = this;

        InitializeComponent();
        var tmp = new PredictionsCanvasViewModel();
        DataContext = tmp;
    }

    private void Button_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        var tmp = drawingCanvas.GetBitmap().ToBlackWhite().CropWhite().Resize(28, 28).RValueToFlatIntArray();

        double[] input = tmp.Select(x => (double)x).ToArray();
        var preditions = App.NeuralNetwork.Predict(input);
        MessageBox.Show($"Predicted number: {Array.IndexOf(preditions, preditions.Max())}");
    }
}
