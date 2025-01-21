using DrawingIdentifierGui.MVVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ToyNeuralNetwork;
using System.Windows;
using ToyNeuralNetwork.Utils;

namespace DrawingIdentifierGui.Models
{
    public class LayerModel : ViewModelBase
    {
        public LayerModel()
        {
            LayerType = LayerType.FullyConnected;
        }

        public IEnumerable<ActivationFunction> ActivationFunctions
        {
            get
            {
                yield return ActivationFunction.ReLU;
                yield return ActivationFunction.Sigmoid;
                yield return ActivationFunction.Softmax;
            }
        }

        public IEnumerable<LayerType> LayerTypes
        {
            get
            {
                yield return LayerType.Convolution;
                yield return LayerType.Pooling;
                yield return LayerType.FullyConnected;
                yield return LayerType.Dropout;
            }
        }

        private LayerType layerType;
        public LayerType LayerType
        {
            get => layerType;
            set
            {
                layerType = value;

                HideAll();

                switch (value)
                {
                    case LayerType.Convolution:
                        IsKernelSizeVisable = Visibility.Visible;
                        IsKernelDepthVisable = Visibility.Visible;
                        IsActivationFunctionVisable = Visibility.Visible;
                        break;

                    case LayerType.Pooling:
                        IsPoolSizeVisable = Visibility.Visible;
                        IsPoolStrideVisable = Visibility.Visible;
                        break;

                    case LayerType.FullyConnected:
                        IsLayerSizeVisable = Visibility.Visible;
                        IsActivationFunctionVisable = Visibility.Visible;
                        break;

                    case LayerType.Dropout:
                        IsDropoutRateVisable = Visibility.Visible;
                        break;
                }

                OnPropertyChanged();
            }
        }

        private void HideAll()
        {
            IsKernelSizeVisable = Visibility.Collapsed;
            IsKernelDepthVisable = Visibility.Collapsed;
            IsPoolSizeVisable = Visibility.Collapsed;
            IsPoolStrideVisable = Visibility.Collapsed;
            IsLayerSizeVisable = Visibility.Collapsed;
            IsActivationFunctionVisable = Visibility.Collapsed;
            IsDropoutRateVisable = Visibility.Collapsed;
        }

        //convo
        private Visibility isKernelSizeVisable;

        public Visibility IsKernelSizeVisable
        {
            get => isKernelSizeVisable;
            set { isKernelSizeVisable = value; OnPropertyChanged(); }
        }

        private int kernelSize;
        public int KernelSize
        {
            get => kernelSize;
            set { kernelSize = value; OnPropertyChanged(); }
        }

        private Visibility isKernelDepthVisable;
        public Visibility IsKernelDepthVisable
        {
            get => isKernelDepthVisable;
            set { isKernelDepthVisable = value; OnPropertyChanged(); }
        }

        private int kernelDepth;
        public int KernelDepth
        {
            get => kernelDepth;
            set { kernelDepth = value; OnPropertyChanged(); }
        }

        //max pooling
        private Visibility isPoolSizeVisable;

        public Visibility IsPoolSizeVisable
        {
            get => isPoolSizeVisable;
            set { isPoolSizeVisable = value; OnPropertyChanged(); }
        }

        private int poolSize;
        public int PoolSize
        {
            get => poolSize;
            set { poolSize = value; OnPropertyChanged(); }
        }

        private Visibility isPoolStrideVisable;
        public Visibility IsPoolStrideVisable
        {
            get => isPoolStrideVisable;
            set { isPoolStrideVisable = value; OnPropertyChanged(); }
        }

        private int poolStride;
        public int PoolStride
        {
            get => poolStride;
            set { poolStride = value; OnPropertyChanged(); }
        }

        //fully connected
        private Visibility isLayerSizeVisable;

        public Visibility IsLayerSizeVisable
        {
            get => isLayerSizeVisable;
            set { isLayerSizeVisable = value; OnPropertyChanged(); }
        }

        private int layerSize;
        public int LayerSize
        {
            get => layerSize;
            set { layerSize = value; OnPropertyChanged(); }
        }

        private Visibility isActivationFunctionVisable;
        public Visibility IsActivationFunctionVisable
        {
            get => isActivationFunctionVisable;
            set { isActivationFunctionVisable = value; OnPropertyChanged(); }
        }

        private ActivationFunction activationFunction;
        public ActivationFunction ActivationFunction
        {
            get => activationFunction;
            set { activationFunction = value; OnPropertyChanged(); }
        }

        //dropout
        private Visibility isDropoutRateVisable;

        public Visibility IsDropoutRateVisable
        {
            get => isDropoutRateVisable;
            set { isDropoutRateVisable = value; OnPropertyChanged(); }
        }

        private float dropoutRate;
        public float DropoutRate
        {
            get => dropoutRate;
            set { dropoutRate = value; OnPropertyChanged(); }
        }
    }
}