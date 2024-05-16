using DrawingIdentifierGui.MVVM;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
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

namespace DrawingIdentifierGui.Views.Controls
{
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

        // Using a DependencyProperty as the backing store for NeuralNetworkType.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty NeuralNetworkTypeProperty =
            DependencyProperty.Register("NeuralNetworkType", typeof(int), typeof(NeuralNetworkOutput), new PropertyMetadata(-1));

        public string HeaderText
        {
            get { return (string)GetValue(HeaderTextProperty); }
            set { SetValue(HeaderTextProperty, value); }
        }

        // Using a DependencyProperty as the backing store for HeaderText.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty HeaderTextProperty =
            DependencyProperty.Register("HeaderText", typeof(string), typeof(NeuralNetworkOutput), new PropertyMetadata("NONE"));

        public NeuralNetworkOutput()
        {
            InitializeComponent();
            this.DataContext = this;
        }

        public void UpdatePrecidtions(double[] nnInput)
        {
            switch (NeuralNetworkType)
            {
                case 0:
                    {
                        var preditions = App.NeuralNetwork.Predict(nnInput);
                        ImagesProcessor.DataReader.SaveToImage(nnInput, "D:\\GoogleDriveMirror\\Studia\\Inzynierka\\text.png");

                        Debug.WriteLine($"[BASE-NN PREDICTION]: {Array.IndexOf(preditions, preditions.Max())}");

                        break;
                    }
                case 1:
                    {
                        //TODO
                        break;
                    }
                default:
                    MessageBox.Show("There is no neural network type: " + NeuralNetworkType);
                    break;
            }
        }
    }
}