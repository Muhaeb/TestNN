using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNN
{
    public class NNException : System.Exception
    {
        public NNException(String message) : base(message)
        {
        }
    }
}
