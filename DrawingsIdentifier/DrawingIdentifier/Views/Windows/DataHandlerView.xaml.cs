using DrawingIdentifierGui.Models;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DrawingIdentifierGui.Views.Windows
{
    /// <summary>
    /// Interaction logic for DataHandlerView.xaml
    /// </summary>
    public partial class DataHandlerView : UserControl
    {
        public DataHandlerView()
        {
            InitializeComponent();
        }

        private void ClassesDataGrid_Loaded(object sender, RoutedEventArgs e)
        {
            var dataGrid = sender as DataGrid;
            if (dataGrid == null) return;

            for (int i = 0; i < App.CLASSES_AMOUNT; i++)
            {
                var bindingPath = $"ImagesCollection[{i}]";
                var column = new DataGridTemplateColumn
                {
                    Header = $"{ToyNeuralNetwork.QuickDrawHandler.QuickDrawSet.IndexToCategory[i]}",
                    CanUserResize = false,
                    IsReadOnly = true,
                    Width = new DataGridLength(1, DataGridLengthUnitType.Star),
                    CanUserReorder = false,
                };

                var templateXaml = $"<DataTemplate xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation'>" +
                                   $"<Image Source='{{Binding {bindingPath}}}' Margin='{1}' /></DataTemplate>";

                column.CellTemplate = (DataTemplate)XamlReader.Parse(templateXaml);
                dataGrid.Columns.Add(column);
            }
        }
    }
}