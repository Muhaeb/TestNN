using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNN
{
    abstract class Layer
    {
        public abstract void Forward();
        public abstract void Backward();
    }
    abstract class Network
    {
        private int numLayers;
        private List<Layer> layers;
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
        }
        public abstract void Forward();
        public abstract void Backward();
        public abstract void Criterion();
    }
}
