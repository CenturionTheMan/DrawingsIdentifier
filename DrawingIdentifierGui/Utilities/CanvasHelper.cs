using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using SystemMedia = System.Windows.Media;
using System.Windows.Media.Imaging;
using System.IO;
using System.Drawing;


namespace DrawingIdentifierGui.Utilities
{
    internal static class CanvasHelper
    {
        public static Bitmap GetBitmap(this InkCanvas inkCanvas)
        {
            int margin = (int)inkCanvas.Margin.Left;
            int width = (int)inkCanvas.ActualWidth - margin;
            int height = (int)inkCanvas.ActualHeight - margin;
            //render ink to bitmap
            RenderTargetBitmap renderBitmap =
            new RenderTargetBitmap(width, height, 96d, 96d, SystemMedia.PixelFormats.Default);
            renderBitmap.Render(inkCanvas);

            //save the ink to a memory stream
            BmpBitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(renderBitmap));

            using (MemoryStream ms = new MemoryStream())
            {
                encoder.Save(ms);
                var bitmap = new Bitmap(ms);
                return new Bitmap(bitmap);
            }
        }
    }
}
