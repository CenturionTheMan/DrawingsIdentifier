using NumSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary.QuickDrawHandler;

public static class QuickDrawDataReader
{
    private static Random random = new Random();

    public static QuickDrawSet LoadQuickDrawSamplesFromFiles(string[] filePaths, int amountToLoadFromEachFile = 2000, bool randomlyShift = true, bool colorReverse = true, float maxValue = 255.0f)
    {
        var files = filePaths;

        Debug.WriteLine($"[LOADING SETS] Found {files.Length} files");

        List<QuickDrawSample> samples = new List<QuickDrawSample>(filePaths.Length * amountToLoadFromEachFile);

        int count = 0;
        foreach (var filePath in files)
        {
            var quickDrawSet = LoadDataFromNpyFile(filePath, amountToLoadFromEachFile, randomlyShift, colorReverse, maxValue);
            samples.AddRange(quickDrawSet);

            count++;
            Debug.WriteLine($"[LOADING SETS] Loaded {count}/{files.Length} files");
        }

        return new QuickDrawSet(samples);
    }

    public static QuickDrawSet LoadQuickDrawSamplesFromDirectory(string directoryPath, int amountToLoadFromEachFile = 2000, bool randomlyShift = true, bool colorReverse = true, float maxValue = 255.0f)
    {
        var files = Directory.GetFiles(directoryPath, "*.npy");

        return LoadQuickDrawSamplesFromFiles(files, amountToLoadFromEachFile, randomlyShift, colorReverse, maxValue);
    }

    private static IEnumerable<QuickDrawSample> LoadDataFromNpyFile(string path, int amountToLoad, bool randomlyShift, bool colorReverse = true, float maxValue = 255.0f)
    {
        NDArray npArray = np.load(path);
        float[,] array = (float[,])npArray.ToMuliDimArray<float>();

        QuickDrawSample[] quickDrawSamples = new QuickDrawSample[amountToLoad];
        string categoryName = Path.GetFileName(path.Replace(".npy", ""));

        int upperBound = amountToLoad > array.GetLength(0) ? array.GetLength(0) : amountToLoad;

        float factor = 1 / maxValue;

        Parallel.For(0, upperBound, i =>
        {
            int sampleIndex = random.Next(array.GetLength(0));
            float[] data = new float[array.GetLength(1)];
            for (int j = 0; j < array.GetLength(1); j++)
            {
                data[j] = array[sampleIndex, j];
            }

            Matrix tmp = MatrixExtender.UnflattenMatrix(new Matrix(data), 28, 28)[0];
            tmp = colorReverse? tmp.ApplyFunction(x => 1 - (x * factor)) : tmp * factor;

            if (randomlyShift)
            {
                float background = colorReverse ? 1.0f : 0.0f;
                tmp = ImageEditor.RandomShiftMatrixImage(tmp, 0.95f, 1.05f, -30f, 30f, background);
            }

            if (tmp.RowsAmount != 28 || tmp.ColumnsAmount != 28)
                throw new Exception("Matrix size is not 28x28");

            quickDrawSamples[i] = new QuickDrawSample(categoryName, [tmp]);
        });

        return quickDrawSamples;
    }
}