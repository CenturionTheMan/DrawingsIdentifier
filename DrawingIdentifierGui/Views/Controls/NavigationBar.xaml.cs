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
    /// Interaction logic for NavigationBar.xaml
    /// </summary>
    public partial class NavigationBar : UserControl
    {
        public ICommand Button1Command
        {
            get { return (ICommand)GetValue(Button1CommandProperty); }
            set { SetValue(Button1CommandProperty, value); }
        }

        public static readonly DependencyProperty Button1CommandProperty =
            DependencyProperty.Register("Button1Command", typeof(ICommand), typeof(NavigationBar), new PropertyMetadata(null));

        public ICommand Button2Command
        {
            get { return (ICommand)GetValue(Button2CommandProperty); }
            set { SetValue(Button2CommandProperty, value); }
        }

        public static readonly DependencyProperty Button2CommandProperty =
            DependencyProperty.Register("Button2Command", typeof(ICommand), typeof(NavigationBar), new PropertyMetadata(null));

        public ICommand Button3Command
        {
            get { return (ICommand)GetValue(Button3CommandProperty); }
            set { SetValue(Button3CommandProperty, value); }
        }

        public static readonly DependencyProperty Button3CommandProperty =
            DependencyProperty.Register("Button3Command", typeof(ICommand), typeof(NavigationBar), new PropertyMetadata(null));

        public ICommand Button4Command
        {
            get { return (ICommand)GetValue(Button4CommandProperty); }
            set { SetValue(Button4CommandProperty, value); }
        }

        public static readonly DependencyProperty Button4CommandProperty =
            DependencyProperty.Register("Button4Command", typeof(ICommand), typeof(NavigationBar), new PropertyMetadata(null));

        public ICommand Button5Command
        {
            get { return (ICommand)GetValue(Button5CommandProperty); }
            set { SetValue(Button5CommandProperty, value); }
        }

        public static readonly DependencyProperty Button5CommandProperty =
            DependencyProperty.Register("Button5Command", typeof(ICommand), typeof(NavigationBar), new PropertyMetadata(null));


        public ICommand Button6Command
        {
            get { return (ICommand)GetValue(Button6CommandProperty); }
            set { SetValue(Button6CommandProperty, value); }
        }

        public static readonly DependencyProperty Button6CommandProperty =
            DependencyProperty.Register("Button6Command", typeof(ICommand), typeof(NavigationBar), new PropertyMetadata(null));


        public NavigationBar()
        {
            InitializeComponent();
        }

        private void Button1_Click(object sender, RoutedEventArgs e)
        {
            Button1Command?.Execute(null);
        }

        private void Button2_Click(object sender, RoutedEventArgs e)
        {
            Button2Command?.Execute(null);
        }

        private void Button3_Click(object sender, RoutedEventArgs e)
        {
            Button3Command?.Execute(null);
        }

        private void Button4_Click(object sender, RoutedEventArgs e)
        {
            Button4Command?.Execute(null);
        }

        private void Button5_Click(object sender, RoutedEventArgs e)
        {
            Button5Command?.Execute(null);
        }

        private void Button6_Click(object sender, RoutedEventArgs e)
        {
            Button6Command?.Execute(null);
        }
        private void NavigationRadioButton_Click(object sender, RoutedEventArgs e)
        {

        }
    }
}