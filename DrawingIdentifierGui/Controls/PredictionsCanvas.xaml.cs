using System.Drawing;
using System.IO;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using static DrawingIdentifierGui.BitmapCustomExtender;

namespace DrawingIdentifierGui.Controls
{
    /// <summary>
    /// Interaction logic for PredictionsCanvas.xaml
    /// </summary>
    public partial class PredictionsCanvas : UserControl
    {
        public PredictionsCanvas()
        {
            InitializeComponent();
        }

        



        private void drawingCanvas_MouseLeave(object sender, MouseEventArgs e)
        {
            System.Drawing.Color color = System.Drawing.Color.Black;

            var bitmap = drawingCanvas.GetBitmap();
            bitmap.Save("D:\\GoogleDriveMirror\\Projects\\DrawingsIdentifier\\tmp1.png");

            var bitmap2 = bitmap.CropByColor(color);
            bitmap2.Save("D:\\GoogleDriveMirror\\Projects\\DrawingsIdentifier\\tmp2.png");
        }


    }
}
