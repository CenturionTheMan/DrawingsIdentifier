using DrawingIdentifierGui.MVVM;
using DrawingIdentifierGui.Views.Windows;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Ink;
using System.Windows.Input;

namespace DrawingIdentifierGui.ViewModels.Windows;

internal class PredictionsCanvasViewModel : ViewModelBase
{
    public RelayCommand PenSelectedCommand => new RelayCommand(PenSelected);
    public RelayCommand EraserSelectedCommand => new RelayCommand(EraserSelected);

    private PredictionsCanvas predictionsCanvas;

    public PredictionsCanvasViewModel()
    {
        predictionsCanvas = PredictionsCanvas.Instance;
    }

    public void PenSelected(object? tmp)
    {
        predictionsCanvas.Cursor = Cursors.Pen;
        predictionsCanvas.drawingCanvas.EditingMode = InkCanvasEditingMode.Ink;
        predictionsCanvas.drawingCanvas.DefaultDrawingAttributes.Color = System.Windows.Media.Colors.Black;
    }

    public void EraserSelected(object? tmp)
    {
        predictionsCanvas.Cursor = Cursors.Cross;
        predictionsCanvas.drawingCanvas.EditingMode = InkCanvasEditingMode.EraseByPoint;
    }
}