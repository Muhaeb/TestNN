using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace TestNN
{
    [Serializable]
    public class Matrix
    {
        
        public double this[int row, int col]
        {
            get
            {
                Validate(row, col);
                return this.matrix[row, col];
            }
            set
            {
                Validate(row, col);
                if (double.IsInfinity(value) || double.IsNaN(value))
                {
                    throw new MathException("Trying to assign invalid number to matrix: "
                            + value);
                }
                this.matrix[row, col] = value;
            }
        }
        
        public static Matrix CreateColumnMatrix(double[] input)
        {
            double[,] d = new double[input.Length, 1];
            for (int row = 0; row < d.Length; row++)
            {
                d[row, 0] = input[row];
            }
            return new Matrix(d);
        }
        public static Matrix CreateColumnMatrix(byte[] input)
        {
            double[,] d = new double[input.Length, 1];
            for (int row = 0; row < d.Length; row++)
            {
                d[row, 0] = (double)input[row];
            }
            return new Matrix(d);
        }
        public static Matrix CreateRowMatrix(double[] input)
        {
            double[,] d = new double[1, input.Length];

            for (int i = 0; i < input.Length; i++)
            {
                d[0, i] = input[i];
            }

            return new Matrix(d);
        }
        public static Matrix CreateRowMatrix(byte[] input)
        {
            double[,] d = new double[1, input.Length];

            for (int i = 0; i < input.Length; i++)
            {
                d[0, i] = (double)input[i];
            }

            return new Matrix(d);
        }

        double[,] matrix;
   
        public Matrix(bool[,] sourceMatrix)
        {

            this.matrix = new double[sourceMatrix.GetUpperBound(0) + 1, sourceMatrix.GetUpperBound(1) + 1];
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    if (sourceMatrix[r, c])
                    {
                        this[r, c] = 1;
                    }
                    else
                    {
                        this[r, c] = -1;
                    }
                }
            }
        }

        public Matrix(byte[,] sourceMatrix)
        {
            this.matrix = new double[sourceMatrix.GetUpperBound(0) + 1, sourceMatrix.GetUpperBound(1) + 1];
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    this[r, c] = (double)sourceMatrix[r, c];
                }
            }
        }

        public Matrix(double[,] sourceMatrix)
        {
            this.matrix = new double[sourceMatrix.GetUpperBound(0) + 1, sourceMatrix.GetUpperBound(1) + 1];
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    this[r, c] = sourceMatrix[r, c];
                }
            }
        }
       
        public Matrix(int rows, int cols)
        {
            this.matrix = new double[rows, cols];
        }

        public Matrix(int rows, int cols, double value)
        {
            this.matrix = new double[rows, cols];
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    this[r, c] = value;
                }
            }
        }
        
        public void Add(int row, int col, double value)
        {
            Validate(row, col);
            double newValue = this[row, col] + value;
            this[row, col] = newValue;
        }

        public void Clear()
        {
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    this[r, c] = 0;
                }
            }
        }
        public Matrix F(string f,double epsilon,bool isDerivative)
        {
            var result = new Matrix(this.Rows, this.Cols);
            for (int row = 0; row < this.Rows; row++)
            {
                for (int col = 0; col < this.Cols; col++)
                {
                    var z = this[row, col];
                    if (f == "Sigmoid")
                        result[row, col] = isDerivative?Utils.dSigmoid(z):Utils.Sigmoid(z);
                    else if (f == "Tanh")
                        result[row, col] = isDerivative?Utils.dTanh(z):Utils.Tanh(z);
                    else if (f == "ReLU")
                        result[row, col] = isDerivative?Utils.dReLU(z):Utils.ReLU(z);
                    else if (f == "LReLU")
                        result[row, col] = isDerivative?Utils.LReLU(z,epsilon):Utils.LReLU(z, epsilon); 
                }
            }
            return result;
        }
       
       
        public Matrix dF(string f)
        {
            var result = new Matrix(this.Rows, this.Cols);
            for (int row = 0; row < this.Rows; row++)
            {
                for (int col = 0; col < this.Cols; col++)
                {
                    var z = this[row, col];
                    if (f == "sigmoid")
                        result[row, col] = Utils.Sigmoid(z);
                    else if (f == "tanh")
                        result[row, col] = (Math.Exp(z) - Math.Exp(-z)) / (Math.Exp(z) - Math.Exp(-z));
                }
            }
            return result;
        }
       
        public Matrix Clone()
        {
            return new Matrix(this.matrix);
        }
        
        public bool Equals(Matrix matrix)
        {
            return equals(matrix, 10);
        }

        public bool equals(Matrix matrix, int precision)
        {

            if (precision < 0)
            {
                throw new MathException("Precision can't be a negative number.");
            }

            double test = Math.Pow(10.0, precision);
            if (double.IsInfinity(test) || (test > long.MaxValue))
            {
                throw new MathException("Precision of " + precision
                        + " decimal places is not supported.");
            }

            precision = (int)Math.Pow(10, precision);

            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    if ((long)(this[r, c] * precision) != (long)(matrix[r, c] * precision))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
       
        public int FromPackedArray(double[] array, int index)
        {

            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    this.matrix[r, c] = array[index++];
                }
            }

            return index;
        }
    
        public Matrix GetCol(int col)
        {
            if (col > this.Cols)
            {
                throw new MathException("Can't get column #" + col
                        + " because it does not exist.");
            }

            double[,] newMatrix = new double[this.Rows, 1];

            for (int row = 0; row < this.Rows; row++)
            {
                newMatrix[row, 0] = this.matrix[row, col];
            }

            return new Matrix(newMatrix);
        }
      
        public int Cols
        {
            get
            {
                return this.matrix.GetUpperBound(1) + 1;
            }
        }
      
        public Matrix GetRow(int row)
        {
            if (row > this.Rows)
            {
                throw new MathException("Can't get row #" + row
                        + " because it does not exist.");
            }

            double[,] newMatrix = new double[1, this.Cols];

            for (int col = 0; col < this.Cols; col++)
            {
                newMatrix[0, col] = this.matrix[row, col];
            }

            return new Matrix(newMatrix);
        }
   
        public int Rows
        {
            get
            {
                return this.matrix.GetUpperBound(0) + 1;
            }
        }


       
        public bool IsVector()
        {
            if (this.Rows == 1)
            {
                return true;
            }
            else
            {
                return this.Cols == 1;
            }
        }
      
        public bool IsZero()
        {
            for (int row = 0; row < this.Rows; row++)
            {
                for (int col = 0; col < this.Cols; col++)
                {
                    if (this.matrix[row, col] != 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }
      
        public void Ramdomize(double min, double max)
        {

            Random rand = new Random();

            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    this.matrix[r, c] = (rand.NextDouble() * (max - min)) + min;
                }
            }
        }

        public int Size
        {
            get
            {
                return this.Rows * this.Cols;
            }
        }

        public double[] ToPackedArray()
        {
            double[] result = new double[this.Rows * this.Cols];

            int index = 0;
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    result[index++] = this.matrix[r, c];
                }
            }

            return result;
        }
        public void RowVectorize()
        {
            var result = new double[this.Rows * this.Cols,1];

            int index = 0;
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    result[index++,0] = this.matrix[r, c];
                }
            }
            this.matrix = result;
        }

        public void ColumnVectorize()
        {
            var result = new double[1, this.Rows * this.Cols];

            int index = 0;
            for (int r = 0; r < this.Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    result[0, index++] = this.matrix[r, c];
                }
            }
            this.matrix = result;
        }

        private void Validate(int row, int col)
        {
            if ((row >= this.Rows) || (row < 0))
            {
                throw new MathException("The row:" + row + " is out of range:"
                        + this.Rows);
            }

            if ((col >= this.Cols) || (col < 0))
            {
                throw new MathException("The col:" + col + " is out of range:"
                        + this.Cols);
            }
        }
        public override string ToString()
        {
            var sb_matrix = new StringBuilder();
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    sb_matrix.Append(matrix[i, j]);
                    sb_matrix.Append('\t');
                }
                sb_matrix.Append(Environment.NewLine);
            }
            return sb_matrix.ToString();
        }
    }
}
