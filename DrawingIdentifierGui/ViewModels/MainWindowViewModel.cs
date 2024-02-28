using DrawingIdentifierGui.MVVM;
using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DrawingIdentifierGui.ViewModels
{
    internal class MainWindowViewModel : ViewModelBase
    {
        private MainWindow mainWindow;
        
        public RelayCommand ExitCommand => new RelayCommand(parameter => mainWindow.Close());
        public RelayCommand MinimalizeCommand => new RelayCommand(parameter => mainWindow.WindowState = System.Windows.WindowState.Minimized);
        


        public MainWindowViewModel(MainWindow mainWindow)
        {
            this.mainWindow = mainWindow;
        }
    }
}
