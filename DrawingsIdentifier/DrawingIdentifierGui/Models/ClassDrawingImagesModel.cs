using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using NeuralNetworkLibrary.Math;

namespace DrawingIdentifierGui.Models
{
    public class ClassDrawingImagesModel
    {
        public ImageSource[] ImagesCollection { get; set; }

        public ClassDrawingImagesModel(NeuralNetworkLibrary.Math.Matrix[] images)
        {
            ImagesCollection = images.Select(i => ConvertMatrixToImageSource(i)).ToArray();
        }

        private ImageSource ConvertMatrixToImageSource(NeuralNetworkLibrary.Math.Matrix matrix)
        {
            var bm = new Bitmap(matrix.ColumnsAmount, matrix.RowsAmount);

            for (int i = 0; i < matrix.RowsAmount; i++)
            {
                for (int j = 0; j < matrix.ColumnsAmount; j++)
                {
                    int rgb = (int)(matrix[i, j] * 255.0f);
                    var color = System.Drawing.Color.FromArgb(rgb, rgb, rgb);
                    bm.SetPixel(j, i, color);
                }
            }

            using (MemoryStream memory = new MemoryStream())
            {
                bm.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }
    }
}