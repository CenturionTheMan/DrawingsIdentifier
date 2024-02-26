using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Drawing.Printing;
using System.Windows.Media.Media3D;
using System.Windows;
using static System.Net.Mime.MediaTypeNames;
using System.Windows.Automation.Peers;

namespace DrawingIdentifierGui;

internal static class BitmapCustomExtender
{
    public static Bitmap GetBitmap(this InkCanvas inkCanvas)
    {
        int margin = (int)inkCanvas.Margin.Left;
        int width = (int)inkCanvas.ActualWidth - margin;
        int height = (int)inkCanvas.ActualHeight - margin;
        //render ink to bitmap
        RenderTargetBitmap renderBitmap =
        new RenderTargetBitmap(width, height, 96d, 96d, PixelFormats.Default);
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

    public static Bitmap CropByColor(this Bitmap bitmap, System.Drawing.Color cropColor)
    {
        int? left = null, right = null, bottom = null, top = null;

        int[,] holder = new int[bitmap.Width, bitmap.Height];

        for (int i = 0; i < bitmap.Width; i++)
        {
            for (int j = 0; j < bitmap.Height; j++)
            {
                var tmp = bitmap.GetPixel(i, j);
                holder[i, j] = tmp.ToArgb();
                if (!(tmp.A == cropColor.A && tmp.B == cropColor.B && tmp.R == cropColor.R)) continue;
                
                if(!bottom.HasValue || j < bottom.Value)
                {
                    bottom = j;
                }

                if (!top.HasValue || j > top.Value)
                {
                    top = j;
                }

                if (!left.HasValue || i < left.Value)
                {
                    left = i;
                }

                if (!right.HasValue || i > right.Value)
                {
                    right = i;
                }
            }
        }

        if (!left.HasValue || !right.HasValue || !bottom.HasValue || !top.HasValue)
            return bitmap;

        return bitmap.Crop(new Rectangle(left.Value, bottom.Value, right.Value - left.Value, top.Value - bottom.Value));
    }

    public static Bitmap Crop(this Bitmap bitmap, Rectangle cropRect)
    {
        Bitmap target = new Bitmap(cropRect.Width, cropRect.Height);
        
        using (Graphics g = Graphics.FromImage(target))
        {
            g.DrawImage(bitmap, new Rectangle(0, 0, target.Width, target.Height),
                cropRect,
                GraphicsUnit.Pixel);
        }

       return target;
    }

    public static Bitmap Resize(this Bitmap bitmap, int width, int height)
    {
        var bmp = new Bitmap(width, height);
        var graph = Graphics.FromImage(bmp);

        // uncomment for higher quality output
        //graph.InterpolationMode = InterpolationMode.High;
        //graph.CompositingQuality = CompositingQuality.HighQuality;
        //graph.SmoothingMode = SmoothingMode.AntiAlias;

        float scale = Math.Min(width / bitmap.Width, height / bitmap.Height);

        var scaleWidth = (int)(bitmap.Width * scale);
        var scaleHeight = (int)(bitmap.Height * scale);

        //var brush = new SolidBrush(System.Drawing.Color.Black);
        //graph.FillRectangle(brush, new RectangleF(0, 0, width, height));

        graph.Clear(System.Drawing.Color.White);
        graph.DrawImage(bitmap, ((int)width - scaleWidth) / 2, ((int)height - scaleHeight) / 2, scaleWidth, scaleHeight);

        return bmp;
    }
}
