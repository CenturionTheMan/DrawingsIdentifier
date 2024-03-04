using System.Drawing;
using System.IO;
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
    public PredictionsCanvas()
    {
        InitializeComponent();
        var tmp = new PredictionsCanvasViewModel(this);
        DataContext = tmp;
    }


}
