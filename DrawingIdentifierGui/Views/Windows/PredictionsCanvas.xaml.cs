using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using static DrawingIdentifierGui.Utilities.CanvasHelper;
using System.Drawing.Printing;
using System.Diagnostics;
using DrawingIdentifierGui.ViewModels.Windows;
using System.Reflection;
using NeuralNetworkLibrary;

namespace DrawingIdentifierGui.Views.Windows;

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
            Matrix mat = new Matrix(bitmap.Height, bitmap.Width);
            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Width; j++)
                {
                    mat[i, j] = bitmap.GetPixel(j, i).R;
                }
            }

            float div = 1 / 255f;
            //TODO CutOffBorderToSquare do not work properly
            var scaled = (mat * div).CutOffBorderToSquare((0.0f, 0.5f))?.ResizeSquare(28, 1f);
            if (scaled == null) return;

            //to remove
            scaled.SaveAsPng("./../../../../UserDrawing.png");


            RunMethodOnCurrentThread(() =>
            {
                NN1Output.UpdatePrecidtions(scaled);
                NN2Output.UpdatePrecidtions(scaled);
            });
        });
        imageTask.Start();
    }

    private void drawingCanvas_MouseLeave(object sender, MouseEventArgs e)
    {
        isDrawing = false;
    }
}