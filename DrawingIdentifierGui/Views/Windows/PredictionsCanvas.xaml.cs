using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using static ImagesProcessor.BitmapCustomExtender;
using static DrawingIdentifierGui.Utilities.CanvasHelper;
using System.Drawing.Printing;
using System.Diagnostics;
using DrawingIdentifierGui.ViewModels.Windows;
using System.Reflection;

namespace DrawingIdentifierGui.Views.Controls;

/// <summary>
/// Interaction logic for PredictionsCanvas.xaml
/// </summary>
public partial class PredictionsCanvas : UserControl
{
    public static PredictionsCanvas Instance;

    private bool isDrawing = false;

    public PredictionsCanvas()
    {
        Instance = this;

        InitializeComponent();
        var tmp = new PredictionsCanvasViewModel();
        DataContext = tmp;
    }

    private void drawingCanvas_PreviewMouseDown(object sender, MouseButtonEventArgs e)
    {
        isDrawing = true;
    }

    private void RunMethodOnCurrentThread(Action action)
    {
        try
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                action();
            });
        }
        catch
        {
        }
    }

    private void drawingCanvas_PreviewMouseMove(object sender, MouseEventArgs e)
    {
        if (isDrawing)
        {
            //on drawing
        }
    }

    private void drawingCanvas_PreviewMouseUp(object sender, MouseButtonEventArgs e)
    {
        isDrawing = false;
        var bitmap = drawingCanvas.GetBitmap();
        var imageTask = new Task(() =>
        {
            var input = bitmap.ToBlackWhite().CropWhite(100).Resize(28, 28).RValueToFlatDoubleArray();

            RunMethodOnCurrentThread(() =>
            {
                BaseNNOutput.UpdatePrecidtions(input);
                ConvolutionalNNOutput.UpdatePrecidtions(input);
            });
        });
        imageTask.Start();
    }

    private void drawingCanvas_MouseLeave(object sender, MouseEventArgs e)
    {
        isDrawing = false;
    }
}