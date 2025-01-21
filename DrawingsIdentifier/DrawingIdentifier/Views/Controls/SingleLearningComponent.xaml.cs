using DrawingIdentifierGui.ViewModels.Controls;
using DrawingIdentifierGui.ViewModels.Windows;
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

namespace DrawingIdentifierGui.Views.Controls
{
    /// <summary>
    /// Interaction logic for SingleLearningComponent.xaml
    /// </summary>
    public partial class SingleLearningComponent : UserControl
    {
        private SingleNetworkLearnigViewModel? viewModel;

        public SingleLearningComponent()
        {
            InitializeComponent();
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            viewModel = new SingleNetworkLearnigViewModel(this.Name);
            DataContext = viewModel;
        }
    }
}