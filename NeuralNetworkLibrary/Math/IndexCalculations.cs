namespace NeuralNetworkLibrary;

static class IndexCalculations
{
    public static int GetIndex(int row, int column, int columnsAmount)
    {
        return row * columnsAmount + column;
    }

    public static (int row, int column) GetRowAndColumn(int index, int columnsAmount)
    {
        return (index / columnsAmount, index % columnsAmount);
    }
}