using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNN
{
    class Utils
    {
        public static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }
        public static double Tanh(double z)
        {
            return (Math.Exp(z) - Math.Exp(-z)) / (Math.Exp(z) + Math.Exp(-z));
        }
        public static double ReLU(double z)
        {
            return (Math.Max(0.0,z));
        }
        public static double LReLU(double z, double alfa)
        {
            return (Math.Max(0, z) + alfa * Math.Min(0, z));
        }

        public static double dSigmoid(double z)
        {
            return Sigmoid(z)*(1 - Sigmoid(z));
        }
        public static double dTanh(double z)
        {
            return (1 - Math.Pow(Tanh(z),2));
        }
        public static double dReLU(double z)
        {
            if (z > 0)
                return 1;
            else
                return 0;
        }
        public static double dLReLU(double z, double alfa)
        {
            if (z > 0)
                return 1;
            else
                return alfa;
        }
    }
}
