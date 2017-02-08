using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNN
{
    public class MatrixMath
    {
        public static Matrix Add(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows)
            {
                throw new MathException(
                        "To add the matrixes they must have the same number of rows and columns.  Matrix a has "
                                + a.Rows
                                + " rows and matrix b has "
                                + b.Rows + " rows.");
            }

            if (a.Cols != b.Cols)
            {
                throw new MathException(
                        "To add the matrixes they must have the same number of rows and columns.  Matrix a has "
                                + a.Cols
                                + " cols and matrix b has "
                                + b.Cols + " cols.");
            }

            double[,] result = new double[a.Rows, a.Cols];

            for (int resultRow = 0; resultRow < a.Rows; resultRow++)
            {
                for (int resultCol = 0; resultCol < a.Cols; resultCol++)
                {
                    result[resultRow, resultCol] = a[resultRow, resultCol]
                            + b[resultRow, resultCol];
                }
            }

            return new Matrix(result);
        }

        public static Matrix F(Matrix a, string f, double epsilon, bool isDerivative)
        {
            var result = new Matrix(a.Rows, a.Cols);
            for (int row = 0; row < a.Rows; row++)
            {
                for (int col = 0; col < a.Cols; col++)
                {
                    var z = a[row, col];
                    if (f == "Sigmoid")
                        result[row, col] = isDerivative ? Utils.dSigmoid(z) : Utils.Sigmoid(z);
                    else if (f == "Tanh")
                        result[row, col] = isDerivative ? Utils.dTanh(z) : Utils.Tanh(z);
                    else if (f == "ReLU")
                        result[row, col] = isDerivative ? Utils.dReLU(z) : Utils.ReLU(z);
                    else if (f == "LReLU")
                        result[row, col] = isDerivative ? Utils.dLReLU(z, epsilon) : Utils.LReLU(z, epsilon);
                }
            }
            return result;
        }

        public static void Copy(Matrix source, Matrix target)
        {
            for (int row = 0; row < source.Rows; row++)
            {
                for (int col = 0; col < source.Cols; col++)
                {
                    target[row, col] = source[row, col];
                }
            }

        }

      
        public static Matrix DeleteCol(Matrix matrix, int deleted)
        {
            if (deleted >= matrix.Cols)
            {
                throw new MathException("Can't delete column " + deleted
                        + " from matrix, it only has " + matrix.Cols
                        + " columns.");
            }
            double[,] newMatrix = new double[matrix.Rows, matrix
                    .Cols - 1];

            for (int row = 0; row < matrix.Rows; row++)
            {
                int targetCol = 0;

                for (int col = 0; col < matrix.Cols; col++)
                {
                    if (col != deleted)
                    {
                        newMatrix[row, targetCol] = matrix[row, col];
                        targetCol++;
                    }

                }

            }
            return new Matrix(newMatrix);
        }

      
        public static Matrix DeleteRow(Matrix matrix, int deleted)
        {
            if (deleted >= matrix.Rows)
            {
                throw new MathException("Can't delete row " + deleted
                        + " from matrix, it only has " + matrix.Rows
                        + " rows.");
            }
            double[,] newMatrix = new double[matrix.Rows - 1, matrix
                    .Cols];
            int targetRow = 0;
            for (int row = 0; row < matrix.Rows; row++)
            {
                if (row != deleted)
                {
                    for (int col = 0; col < matrix.Cols; col++)
                    {
                        newMatrix[targetRow, col] = matrix[row, col];
                    }
                    targetRow++;
                }
            }
            return new Matrix(newMatrix);
        }

        public static double SumExp(Matrix a)
        {
            double result = 0;
            for (int row = 0; row < a.Rows; row++)
            {
                for (int col = 0; col < a.Cols; col++)
                {
                    result += Math.Exp(a[row, col]);
                }
            }
            return result;
        }
        public static double PNorm(Matrix a,int p)
        {
            if(p < 1)
            {
                throw new MathException("p-norm can only be computed for p >= 1");
            }
            double result = 0;
            for (int r = 0; r < a.Rows; r++)
            {
                for (int c = 0; c < a.Cols; c++)
                {
                    result += Math.Pow(a[r, c],p);
                }
            }
            return Math.Pow(result,1.0/p);
        }
        public static double Sum(Matrix a)
        {
            double result = 0;
            for (int r = 0; r < a.Rows; r++)
            {
                for (int c = 0; c < a.Cols; c++)
                {
                    result += a[r, c];
                }
            }
            return result;
        }
        public static double Trace(Matrix a)
        {
            if (a.Rows != a.Cols)
                throw new MathException("Trace Can be computed for square matrices only");
            double result = 0;
            for (int r = 0; r < a.Rows; r++)
            {
                result += a[r, r];
            }
            return result;
        }
        public static Matrix Divide(Matrix a, double b)
        {
            double[,] result = new double[a.Rows, a.Cols];
            for (int row = 0; row < a.Rows; row++)
            {
                for (int col = 0; col < a.Cols; col++)
                {
                    result[row, col] = a[row, col] / b;
                }
            }
            return new Matrix(result);
        }

       
        public static double DotProduct(Matrix a, Matrix b)
        {
            if (!a.IsVector() || !b.IsVector())
            {
                throw new MathException(
                        "To take the dot product, both matrixes must be vectors.");
            }

            Double[] aArray = a.ToPackedArray();
            Double[] bArray = b.ToPackedArray();

            if (aArray.Length != bArray.Length)
            {
                throw new MathException(
                        "To take the dot product, both matrixes must be of the same length.");
            }

            double result = 0;
            int length = aArray.Length;

            for (int i = 0; i < length; i++)
            {
                result += aArray[i] * bArray[i];
            }

            return result;
        }
        public static Matrix Identity(int size)
        {
            if (size < 1)
            {
                throw new MathException("Identity matrix must be at least of size 1.");
            }

            Matrix result = new Matrix(size, size);

            for (int i = 0; i < size; i++)
            {
                result[i, i] = 1;
            }

            return result;
        }
        public static Tuple<int,int,double> Max(Matrix a)
        {
            double maxValue = 0;
            int maxRow = 0;
            int maxCol = 0;
            for (int row = 0; row < a.Rows; row++)
            {
                for (int col = 0; col < a.Cols; col++)
                {
                    if (Math.Abs(a[row, col]) > maxValue)
                    {
                        maxValue = Math.Abs(a[row, col]);
                        maxRow = row;
                        maxCol = col;
                    }
                }
            }
            return new Tuple<int,int,double>(maxRow,maxCol,maxValue);
        }
        public static Matrix Multiply(Matrix a, double b)
        {
            double[,] result = new double[a.Rows, a.Cols];
            for (int row = 0; row < a.Rows; row++)
            {
                for (int col = 0; col < a.Cols; col++)
                {
                    result[row, col] = a[row, col] * b;
                }
            }
            return new Matrix(result);
        }
        public static Matrix HadamardProduct(Matrix a, Matrix b)
        {
            if (a.Cols != b.Cols && a.Rows != b.Rows)
            {
                throw new MathException(
                        "To use hadamard matrix multiplication the sizes of the matrices/vectors must match.");
            }
            double[,] result = new double[a.Rows, a.Cols];
            for (int row = 0; row < a.Rows; row++)
            {
                for (int col = 0; col < a.Cols; col++)
                {
                    result[row, col] = a[row, col] * b[row, col];
                }
            }
            return new Matrix(result);
        }

        public static Matrix Multiply(Matrix a, Matrix b)
        {
            if (a.Cols != b.Rows)
            {
                throw new MathException(
                        "To use ordinary matrix multiplication the number of columns on the first matrix must mat the number of rows on the second.");
            }

            double[,] result = new double[a.Rows, b.Cols];

            for (int resultRow = 0; resultRow < a.Rows; resultRow++)
            {
                for (int resultCol = 0; resultCol < b.Cols; resultCol++)
                {
                    double value = 0;

                    for (int i = 0; i < a.Cols; i++)
                    {

                        value += a[resultRow, i] * b[i, resultCol];
                    }
                    result[resultRow, resultCol] = value;
                }
            }

            return new Matrix(result);
        }

      
        public static Matrix Subtract(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows)
            {
                throw new MathException(
                        "To subtract the matrixes they must have the same number of rows and columns.  Matrix a has "
                                + a.Rows
                                + " rows and matrix b has "
                                + b.Rows + " rows.");
            }

            if (a.Cols != b.Cols)
            {
                throw new MathException(
                        "To subtract the matrixes they must have the same number of rows and columns.  Matrix a has "
                                + a.Cols
                                + " cols and matrix b has "
                                + b.Cols + " cols.");
            }

            double[,] result = new double[a.Rows, a.Cols];

            for (int resultRow = 0; resultRow < a.Rows; resultRow++)
            {
                for (int resultCol = 0; resultCol < a.Cols; resultCol++)
                {
                    result[resultRow, resultCol] = a[resultRow, resultCol]
                            - b[resultRow, resultCol];
                }
            }

            return new Matrix(result);
        }

       
        public static Matrix Transpose(Matrix input)
        {
            double[,] inverseMatrix = new double[input.Cols, input
                    .Rows];

            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Cols; c++)
                {
                    inverseMatrix[c, r] = input[r, c];
                }
            }

            return new Matrix(inverseMatrix);
        }
    }
}
