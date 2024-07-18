﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary.QuickDrawHandler;

public struct QuickDrawSample
{
    public readonly string category;
    public readonly Matrix[] data;

    public QuickDrawSample(string category, Matrix[] data)
    {
        this.category = category;
        this.data = data;
    }
}

public class QuickDrawSet
{
    private static Random random = new();

    public static readonly Dictionary<string, int> CategoryToIndex = new Dictionary<string, int>{
        { "axe", 0},
        { "cactus", 1},
        { "cat", 2},
        { "diamond", 3},
        { "fence", 4},
        { "moustache", 5},
        { "pants", 6},
        { "snowman", 7},
        { "stairs", 8},
        { "sword", 9},
    };

    public static readonly Dictionary<int, string> IndexToCategory = CategoryToIndex.ToDictionary(x => x.Value, x => x.Key);

    public readonly IEnumerable<QuickDrawSample> samples;

    public QuickDrawSet(IEnumerable<QuickDrawSample> samples, bool shuffleData = true)
    {
        if (shuffleData)
            this.samples = samples.OrderBy(x => random.Next());
        else
            this.samples = samples;
    }

    private Matrix OutputForNN(string category)
    {
        float[] output = new float[10];
        output[CategoryToIndex[category]] = 1;

        return new Matrix(output);
    }

    public ((Matrix[] inputs, Matrix outputs)[] trainData, (Matrix[] inputs, Matrix outputs)[] testData) SplitIntoTrainTest(int testSizePercent = 20)
    {
        int testCount = (int)(samples.Count() * (testSizePercent / 100.0));
        var shuffledData = samples.OrderBy(x => Guid.NewGuid()).ToList();

        IEnumerable<QuickDrawSample> trainData = shuffledData.Skip(testCount);
        IEnumerable<QuickDrawSample> testData = shuffledData.Take(testCount);

        var train = trainData.Select(x => (x.data, OutputForNN(x.category))).ToArray();
        var test = testData.Select(x => (x.data, OutputForNN(x.category))).ToArray();

        return (train, test);
    }

    public ((Matrix[] inputs, Matrix outputs)[] trainData, (Matrix[] inputs, Matrix outputs)[] testData) SplitIntoTrainTestFlattenInput(int testSizePercent = 20)
    {
        (var trainData, var testData) = SplitIntoTrainTest(testSizePercent);
        var trainFlat = trainData.Select(i => (new Matrix[] { MatrixExtender.FlattenMatrix(i.inputs) }, i.outputs));
        var testFlat = testData.Select(i => (new Matrix[] { MatrixExtender.FlattenMatrix(i.inputs) }, i.outputs));

        return (trainFlat.ToArray(), testFlat.ToArray());
    }
}