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

public class PredictionsCanvasViewModel : ViewModelBase
{
    public RelayCommand PenSelectedCommand => new RelayCommand(PenSelected);
    public RelayCommand EraserSelectedCommand => new RelayCommand(EraserSelected);
    public RelayCommand ClearCanvasCommand => new RelayCommand(ClearCanvas);

    private PredictionsCanvas predictionsCanvas;

    public PredictionsCanvasViewModel()
    {
        predictionsCanvas = PredictionsCanvas.Instance;
    }

    public void ClearCanvas(object? tmp)
    {
        predictionsCanvas.drawingCanvas.Strokes.Clear();
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
        //predictionsCanvas.drawingCanvas.EraserShape
        predictionsCanvas.drawingCanvas.EditingMode = InkCanvasEditingMode.EraseByPoint;
        predictionsCanvas.drawingCanvas.DefaultDrawingAttributes.Height = 40;
        predictionsCanvas.drawingCanvas.DefaultDrawingAttributes.Width = 40;
    }
}