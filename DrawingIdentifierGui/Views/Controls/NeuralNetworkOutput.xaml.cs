using NeuralNetworkLibrary;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Matrix = NeuralNetworkLibrary.Matrix;

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

    public void UpdatePrecidtions(Matrix mat)
    {
        Matrix predition = App.NeuralNetworks[NeuralNetworkType].Predict(mat);

        for (int i = 0; i < singleNNNodes.Length; i++)
        {
            singleNNNodes[i].SetPredictionValue(predition[i, 0], DefaultNodeBg);
        }

        singleNNNodes[predition.IndexOfMax()].ActivateBest(ActiveNodeBg);
    }
}