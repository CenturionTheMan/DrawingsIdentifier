using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

using System;

using System.Drawing;
using NumSharp;

using System.Collections.Generic;

using System.Diagnostics;

namespace ImagesProcessor;

public class DataReader
{
    private static Random random = new Random();

    public static QuickDrawSet LoadQuickDrawSamplesFromFiles(string[] filePaths, int amountToLoadFromEachFile = 2000, bool colorReverse = true, float maxValue = 255.0f)
    {
        List<QuickDrawSet> result = new();

        var files = filePaths;

        Debug.WriteLine($"[LOADING SETS] Found {files.Length} files");

        List<QuickDrawSample> samples = new List<QuickDrawSample>();

        int count = 0;
        foreach (var filePath in files)
        {
            var quickDrawSet = LoadDataFromNpyFile(filePath, amountToLoadFromEachFile, colorReverse, maxValue);
            samples.AddRange(quickDrawSet);

            count++;
            Debug.WriteLine($"[LOADING SETS] Loaded {count}/{files.Length} files");
        }

        return new QuickDrawSet(samples);
    }

    public static QuickDrawSet LoadQuickDrawSamplesFromDirectory(string directoryPath, int amountToLoadFromEachFile = 2000, bool colorReverse = true, float maxValue = 255.0f)
    {
        var files = Directory.GetFiles(directoryPath, "*.npy");

        return LoadQuickDrawSamplesFromFiles(files, amountToLoadFromEachFile, colorReverse, maxValue);
    }

    private static IEnumerable<QuickDrawSample> LoadDataFromNpyFile(string path, int amountToLoad, bool colorReverse = true, float maxValue = 255.0f)
    {
        NDArray npArray = np.load(path);
        float[,] array = (float[,])npArray.ToMuliDimArray<float>();

        List<QuickDrawSample> result = new(amountToLoad);
        string categoryName = Path.GetFileName(path.Replace(".npy", ""));

        int upperBound = amountToLoad > array.GetLength(0) ? array.GetLength(0) : amountToLoad;
        Parallel.For(0, upperBound, i =>
        {
            int sampleIndex = random.Next(array.GetLength(0));
            float[] row = new float[array.GetLength(1)];
            for (int j = 0; j < array.GetLength(1); j++)
            {
                row[j] = colorReverse ? 1 - array[sampleIndex, j] / maxValue : array[sampleIndex, j] / maxValue;
            }
            result.Add(new QuickDrawSample(categoryName, row));
        });

        return result;
    }

    public static void SaveToImage(float[,] data, string path)
    {
        var bitmap = new Bitmap(data.GetLength(1), data.GetLength(0));

        for (int y = 0; y < data.GetLength(0); y++)
        {
            for (int x = 0; x < data.GetLength(1); x++)
            {
                float value = data[y, x];

                int colorValue = (int)(value * 255);

                colorValue = Math.Clamp(colorValue, 0, 255);

                Color color = Color.FromArgb(colorValue, colorValue, colorValue);

                bitmap.SetPixel(x, y, color);
            }
        }

        try
        {
            bitmap.Save(path, System.Drawing.Imaging.ImageFormat.Png);
            
        }
        catch (System.Exception)
        {
        }
    }

    public static void SaveToImage(float[] data, string path, int width = 28, int height = 28)
    {
        var bitmap = new Bitmap(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int dataIndex = y * width + x;

                float value = data[dataIndex];

                int colorValue = (int)(value * 255);

                colorValue = Math.Clamp(colorValue, 0, 255);

                Color color = Color.FromArgb(colorValue, colorValue, colorValue);

                bitmap.SetPixel(x, y, color);
            }
        }

        bitmap.Save(path, System.Drawing.Imaging.ImageFormat.Png);
    }
}