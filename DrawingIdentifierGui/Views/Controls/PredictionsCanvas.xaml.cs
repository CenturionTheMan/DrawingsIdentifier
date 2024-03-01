using System.Drawing;
using System.IO;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using DrawingIdentifierGui.Utilities;
using static DrawingIdentifierGui.Utilities.BitmapCustomExtender;

namespace DrawingIdentifierGui.Views.Controls;

/// <summary>
/// Interaction logic for PredictionsCanvas.xaml
/// </summary>
public partial class PredictionsCanvas : UserControl
{
    public PredictionsCanvas()
    {
        InitializeComponent();
    }



    private void Button_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        System.Drawing.Color color = System.Drawing.Color.Black;

        var bitmap = drawingCanvas
            .GetBitmap()
            .CropWhite()
            .Resize(28, 28)
            .ToBlackWhite(reverse: true);
        bitmap.Save("D:\\GoogleDriveMirror\\Projects\\DrawingsIdentifier\\tmp1.png");

      
    }
}
