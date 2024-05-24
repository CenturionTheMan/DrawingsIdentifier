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
    public static QuickDrawSet LoadQuickDrawSamplesFromFiles(string[] filePaths, int amountToLoadFromEachFile = 2000, bool colorReverse = true, double maxValue = 255.0)
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

    public static QuickDrawSet LoadQuickDrawSamplesFromDirectory(string directoryPath, int amountToLoadFromEachFile = 2000, bool colorReverse = true, double maxValue = 255.0)
    {
        var files = Directory.GetFiles(directoryPath, "*.npy");

        return LoadQuickDrawSamplesFromFiles(files, amountToLoadFromEachFile, colorReverse, maxValue);
    }

    private static IEnumerable<QuickDrawSample> LoadDataFromNpyFile(string path, int amountToLoad = 2000, bool colorReverse = true, double maxValue = 255.0)
    {
        NDArray npArray = np.load(path);
        double[,] array = (double[,])npArray.ToMuliDimArray<double>();

        List<QuickDrawSample> result = new(amountToLoad);
        string categoryName = Path.GetFileName(path.Replace(".npy", ""));

        int upperBound = amountToLoad > array.GetLength(0) ? array.GetLength(0) : amountToLoad;
        Parallel.For(0, upperBound, i =>
        {
            double[] row = new double[array.GetLength(1)];
            for (int j = 0; j < array.GetLength(1); j++)
            {
                row[j] = colorReverse ? 1 - array[i, j] / maxValue : array[i, j] / maxValue;
            }
            result.Add(new QuickDrawSample(categoryName, row));
        });

        return result;
    }

    public static void SaveToImage(double[] data, string path, int width = 28, int height = 28)
    {
        var bitmap = new Bitmap(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int dataIndex = y * width + x;

                double value = data[dataIndex];

                int colorValue = (int)(value * 255);

                Color color = Color.FromArgb(colorValue, colorValue, colorValue);

                bitmap.SetPixel(x, y, color);
            }
        }

        bitmap.Save(path, System.Drawing.Imaging.ImageFormat.Png);
    }
}