using System;
using System.Collections.Generic;
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

namespace DrawingIdentifierGui.Views.Windows
{
    /// <summary>
    /// Interaction logic for NeuralNetworkLearning.xaml
    /// </summary>
    public partial class NeuralNetworkLearning : UserControl
    {
        public NeuralNetworkLearning()
        {
            InitializeComponent();
            DataContext = null;
        }

        private void Convolutional_Neural_Network_Loaded(object sender, RoutedEventArgs e)
        {
        }
    }
}