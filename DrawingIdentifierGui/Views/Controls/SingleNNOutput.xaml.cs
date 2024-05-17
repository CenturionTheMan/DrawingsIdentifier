using DrawingIdentifierGui.MVVM;
using System;
using System.Collections.Generic;

//using System.Drawing;
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
    /// Interaction logic for SingleNNOutput.xaml
    /// </summary>
    public partial class SingleNNOutput : UserControl
    {
        public string Text
        {
            get { return (string)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }

        public static readonly DependencyProperty TextProperty =
            DependencyProperty.Register("Text", typeof(string), typeof(SingleNNOutput), new PropertyMetadata("NONE"));

        public Brush CustomBackground
        {
            get { return (Brush)GetValue(CustomBackgroundProperty); }
            set { SetValue(CustomBackgroundProperty, value); }
        }

        public static readonly DependencyProperty CustomBackgroundProperty =
            DependencyProperty.Register("CustomBackground", typeof(Brush), typeof(SingleNNOutput), new PropertyMetadata(Brushes.Gray));

        public string Probability
        {
            get { return (string)GetValue(ProbabilityProperty); }
            set { SetValue(ProbabilityProperty, value); }
        }

        public static readonly DependencyProperty ProbabilityProperty =
            DependencyProperty.Register("Probability", typeof(string), typeof(SingleNNOutput), new PropertyMetadata("???.??%"));

        public SingleNNOutput()
        {
            InitializeComponent();
        }

        public void SetPredictionValue(double probability, Brush defaultBg)
        {
            this.CustomBackground = defaultBg;
            this.Probability = $"{Math.Round((100 * probability), 2)}%";
        }

        public void ActivateBest(Brush background)
        {
            this.CustomBackground = background;
        }
    }
}