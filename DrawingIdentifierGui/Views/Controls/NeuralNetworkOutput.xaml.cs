using DrawingIdentifierGui.MVVM;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DrawingIdentifierGui.Views.Controls;

/// <summary>
/// Interaction logic for NeuralNetworkOutput.xaml
/// </summary>
public partial class NeuralNetworkOutput : UserControl
{
    public int NeuralNetworkType
    {
        get { return (int)GetValue(NeuralNetworkTypeProperty); }
        set { SetValue(NeuralNetworkTypeProperty, value); }
    }

    public static readonly DependencyProperty NeuralNetworkTypeProperty =
        DependencyProperty.Register("NeuralNetworkType", typeof(int), typeof(NeuralNetworkOutput), new PropertyMetadata(-1));

    public string HeaderText
    {
        get { return (string)GetValue(HeaderTextProperty); }
        set { SetValue(HeaderTextProperty, value); }
    }

    public static readonly DependencyProperty HeaderTextProperty =
        DependencyProperty.Register("HeaderText", typeof(string), typeof(NeuralNetworkOutput), new PropertyMetadata("NONE"));

    public Brush DefaultNodeBg
    {
        get { return (Brush)GetValue(DefaultNodeBgProperty); }
        set { SetValue(DefaultNodeBgProperty, value); }
    }

    public static readonly DependencyProperty DefaultNodeBgProperty =
        DependencyProperty.Register("DefaultNodeBgProperty", typeof(Brush), typeof(NeuralNetworkOutput), null);

    public Brush ActiveNodeBg
    {
        get { return (Brush)GetValue(ActiveNodeBgProperty); }
        set { SetValue(ActiveNodeBgProperty, value); }
    }

    public static readonly DependencyProperty ActiveNodeBgProperty =
        DependencyProperty.Register("ActiveNodeBgProperty", typeof(Brush), typeof(NeuralNetworkOutput), new PropertyMetadata(Brushes.Yellow));

    private SingleNNOutput[] singleNNNodes;

    public NeuralNetworkOutput()
    {
        InitializeComponent();
        //this.DataContext = this;

        singleNNNodes = this.HolderGrid.Children.OfType<SingleNNOutput>().ToArray();
    }

    public void UpdatePrecidtions(double[] nnInput)
    {
        double[]? preditions = null;

        switch (NeuralNetworkType)
        {
            case 0:
                {
                    preditions = App.NeuralNetwork.Predict(nnInput);
                    //ImagesProcessor.DataReader.SaveToImage(nnInput, "D:\\GoogleDriveMirror\\Studia\\Inzynierka\\text.png");

                    Debug.WriteLine($"[BASE-NN PREDICTION]: {Array.IndexOf(preditions, preditions.Max())}");

                    break;
                }
            case 1:
                {
                    //TODO
                    break;
                }
            default:
                {
                    MessageBox.Show("There is no neural network type: " + NeuralNetworkType);
                    return;
                }
        }

        if (preditions == null)
            return;

        for (int i = 0; i < singleNNNodes.Length; i++)
        {
            singleNNNodes[i].SetPredictionValue(preditions[i], DefaultNodeBg);
        }

        double max = preditions.Max();
        singleNNNodes[Array.IndexOf(preditions, max)].ActivateBest(ActiveNodeBg);
    }
}