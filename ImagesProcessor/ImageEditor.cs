using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImagesProcessor;

public static class ImageEditor
{
    public static bool SaveFloatArrayAsImage(this float[,] data, string path)
    {
        int height = data.GetLength(0);
        int width = data.GetLength(1);

        using (var image = new Image<Rgba32>(width, height))
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Normalize the float value to a range suitable for image representation (0-255)
                    // This assumes the float values are normalized between 0 and 1
                    byte value = (byte)(data[y, x] * 255);
                    image[x, y] = new Rgba32(value, value, value, 255); // Grayscale value
                }
            }

            // Save the image to the specified path
            try
            {
                image.Save(path);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred while saving the image: {ex.Message}");
                return false;
            }
        }
    }
}