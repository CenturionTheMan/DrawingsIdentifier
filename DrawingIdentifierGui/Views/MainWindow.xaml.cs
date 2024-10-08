using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.ViewModels;
using System.Runtime.CompilerServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DrawingIdentifierGui
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public static MainWindow? Instance;

        public MainWindow()
        {
            Instance = this;

            InitializeComponent();
            this.DataContext = new MainWindowViewModel();
        }

        private void Window_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            this.DragMove();
        }
        

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            if (App.IsExampleNN1Loaded == false && App.IsExampleNN2Loaded == false)
            {
                MessageBox.Show("App was unable to load example neural networks. Not trained networks are being used ...");
            }
            else if (App.IsExampleNN1Loaded == false)
            {
                MessageBox.Show("App was unable to load example neural network 1. Not trained network is being used ...");
            }
            else if (App.IsExampleNN2Loaded == false)
            {
                MessageBox.Show("App was unable to load example neural network 2. Not trained network is being used ...");
            }
            else
            {
                MessageBox.Show("App loaded example neural networks");
            }
        }
    }
}