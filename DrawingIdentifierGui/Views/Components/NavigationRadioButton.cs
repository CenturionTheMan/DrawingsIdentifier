using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace DrawingIdentifierGui.Views.Components;

public class NavigationRadioButton : RadioButton
{


    public int ImageMargin
    {
        get { return (int)GetValue(ImageMarginProperty); }
        set { SetValue(ImageMarginProperty, value); }
    }

    // Using a DependencyProperty as the backing store for ImageMargin.  This enables animation, styling, binding, etc...
    public static readonly DependencyProperty ImageMarginProperty =
        DependencyProperty.Register("ImageMargin", typeof(int), typeof(NavigationRadioButton), new PropertyMetadata(0));



    public ImageSource ImageSource
    {
        get { return (ImageSource)GetValue(ImageSourceProperty); }
        set { SetValue(ImageSourceProperty, value); }
    }

    // Using a DependencyProperty as the backing store for ImageSource.  This enables animation, styling, binding, etc...
    public static readonly DependencyProperty ImageSourceProperty =
        DependencyProperty.Register("ImageSource", typeof(ImageSource), typeof(NavigationRadioButton), new PropertyMetadata(null));





}
