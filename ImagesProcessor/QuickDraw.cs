using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImagesProcessor;

public struct QuickDrawSample
{
    public readonly string category;
    public readonly double[] data;

    public QuickDrawSample(string category, double[] data)
    {
        this.category = category;
        this.data = data;
    }
}

public class QuickDrawSet
{
    public static readonly Dictionary<string, int> categories = new Dictionary<string, int>{
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

    public readonly IEnumerable<QuickDrawSample> samples;

    public QuickDrawSet(IEnumerable<QuickDrawSample> samples, bool shuffleData=true)
    {
        if(shuffleData)
            this.samples = samples.OrderBy(x => Guid.NewGuid());
        else
            this.samples = samples;
    }

    private double[] OutputForNN(string category)
    {
        double[] output = new double[10];
        output[categories[category]] = 1;
        return output;
    }

    public ((double[] inputs, double[] outputs)[] trainData, (double[] inputs, double[] outputs)[] testData) SplitIntoTrainTest(int testSizePercent = 20)
    {
        int testCount = (int)(samples.Count() * (testSizePercent/100.0));
        var shuffledData = samples.OrderBy(x => Guid.NewGuid()).ToList();

        IEnumerable<QuickDrawSample> trainData = shuffledData.Skip(testCount);
        IEnumerable<QuickDrawSample> testData = shuffledData.Take(testCount);

        var train = trainData.Select(x => (x.data, OutputForNN(x.category))).ToArray();
        var test = testData.Select(x => (x.data, OutputForNN(x.category))).ToArray();

        return (train, test);
    }
}
