using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNN
{
    class MathException : System.Exception
    {
        public MathException(String message) : base(message)
        {
        }
    }
}
