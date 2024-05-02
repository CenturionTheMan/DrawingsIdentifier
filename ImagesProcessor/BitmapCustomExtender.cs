using System.Drawing;
using System.IO;

namespace ImagesProcessor;

public static class BitmapCustomExtender
{
    

    public static Bitmap ToBlackWhite(this Bitmap bitmap, int whiteThreshold = 250, bool reverse = false)
    {
        Bitmap image = new Bitmap(bitmap);

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                Color c = image.GetPixel(x, y);
                int luma = (int)(c.R * 0.3 + c.G * 0.59 + c.B * 0.11);

                luma = luma >= whiteThreshold ? 255 : luma;

                if (reverse)
                {
                    luma = 255 - luma;
                }

                image.SetPixel(x, y, Color.FromArgb(luma, luma, luma));
            }
        }

        return image;
    }

    public static Bitmap CropWhite(this Bitmap bitmap, int margin = 20, int whiteThreshold = 250)
    {
        int? left = null, right = null, bottom = null, top = null;


        for (int i = 0; i < bitmap.Width; i++)
        {
            for (int j = 0; j < bitmap.Height; j++)
            {
                var pixel = bitmap.GetPixel(i, j);

                bool isWhite = pixel.R >= whiteThreshold && pixel.G >= whiteThreshold && pixel.B >= whiteThreshold;
                if (isWhite) continue;

                if (!bottom.HasValue || j < bottom.Value)
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

        return bitmap.Crop(new Rectangle(
            left.Value - margin,
            bottom.Value - margin,
            right.Value - left.Value + 2 * margin,
            top.Value - bottom.Value + 2 * margin));
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

    public static int[] RValueToFlatIntArray(this Bitmap bitmap)
    {
        int[] result = new int[bitmap.Width * bitmap.Height];
        int index = 0;
        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                result[index++] = bitmap.GetPixel(x, y).R;
            }
        }
        return result;
    }

    public static Bitmap Resize(this Bitmap bitmap, int width, int height)
    {
        var bmp = new Bitmap(width, height);
        var graphics = Graphics.FromImage(bmp);

        // uncomment for higher quality output
        //graph.InterpolationMode = InterpolationMode.High;
        //graph.CompositingQuality = CompositingQuality.HighQuality;
        //graph.SmoothingMode = SmoothingMode.AntiAlias;

        //float scale = Math.Min(width / bitmap.Width, height / bitmap.Height);

        //var scaleWidth = (int)(bitmap.Width * scale);
        //var scaleHeight = (int)(bitmap.Height * scale);

        //var brush = new SolidBrush(System.Drawing.Color.Black);
        //graph.FillRectangle(brush, new RectangleF(0, 0, width, height));

        graphics.Clear(System.Drawing.Color.White);
        graphics.DrawImage(bitmap, 0, 0, width, height);
        graphics.Dispose();

        return bmp;
    }

}
