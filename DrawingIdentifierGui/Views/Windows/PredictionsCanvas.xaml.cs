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
                    mat[i, j] = bitmap.GetPixel(i, j).R;
                }
            }

            //to remove
            mat.SaveAsPng("./../../../../UserDrawing.png");

            RunMethodOnCurrentThread(() =>
            {
                NN1Output.UpdatePrecidtions(mat);
                NN2Output.UpdatePrecidtions(mat);
            });
        });
        imageTask.Start();
    }

    private void drawingCanvas_MouseLeave(object sender, MouseEventArgs e)
    {
        isDrawing = false;
    }
}