using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace DrawingIdentifierGui.Components;

public class PrimaryButton : Button
{
    public SolidColorBrush DisableBackgroundColour
    {
        get { return (SolidColorBrush)GetValue(DisableBackgroundColourProperty); }
        set { SetValue(DisableBackgroundColourProperty, value); }
    }

    public static readonly DependencyProperty DisableBackgroundColourProperty =
        DependencyProperty.Register("DisableBackgroundColour", typeof(SolidColorBrush), typeof(PrimaryButton), new PropertyMetadata(null));

    public SolidColorBrush DisableForegroundColour
    {
        get { return (SolidColorBrush)GetValue(DisableForegroundColourProperty); }
        set { SetValue(DisableForegroundColourProperty, value); }
    }

    public static readonly DependencyProperty DisableForegroundColourProperty =
        DependencyProperty.Register("DisableForegroundColour", typeof(SolidColorBrush), typeof(PrimaryButton), new PropertyMetadata(null));

    public SolidColorBrush DisableBorderColour
    {
        get { return (SolidColorBrush)GetValue(DisableBorderColourProperty); }
        set { SetValue(DisableBorderColourProperty, value); }
    }

    public static readonly DependencyProperty DisableBorderColourProperty =
        DependencyProperty.Register("DisableBorderColour", typeof(SolidColorBrush), typeof(PrimaryButton), new PropertyMetadata(null));

    public CornerRadius Radius
    {
        get { return (CornerRadius)GetValue(RadiusProperty); }
        set { SetValue(RadiusProperty, value); }
    }

    public static readonly DependencyProperty RadiusProperty =
        DependencyProperty.Register("Radius", typeof(CornerRadius), typeof(PrimaryButton), new PropertyMetadata(new CornerRadius(5)));
}