using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using System.IO;

namespace TestNN
{
    class Program
    {
        static string trainImagesFile;
        static string trainLabelsFile;
        static string testImagesFile;
        static string testLabelsFile;
        static int trainingSize;
        static int testingSize;
        static string logFile;
        static int batchSize;
        static int epochs;
        static double learningRate;
        static double weightDecay;
        static int seed;
        static string[] layerFunctions;  //Sigmoid, Tanh, ReLU, or LReLU (Leaky ReLU) for now
        static string problemType;  //classification or regression
        static string criterion;  //mse or softmax
        static double epsilonLeaky;  //threshold for Leaky ReLU
        static double epsilonGradientCheck;
        static double gradientCheckThreshold;
        static int L;
        static int[] layerSizes;
        static Matrix[] W;
        static Matrix[] dW;
        static List<Matrix[]> acc_dW;
        static Matrix[] deltaW;
        static Matrix[] b;
        static Matrix[] db;
        static Matrix[] deltab;
        static List<Matrix[]> acc_db;
        static Matrix[] errors;
        static double[] Loss;
        static Matrix[] A;
        static Matrix[] Z;
        static Matrix label;
        static Tuple<List<DigitImage>, List<DigitImage>> Data;
        static List<DigitImage> Training;
        static List<DigitImage> Testing;
        static bool CheckGradients(int k, int i)
        {
            bool isWeightGradientCorrect = true;
            bool isBiasGradientCorrect = true;
            double[] diffs = new double[4];
            var Loss_bak = Loss[i];
            Matrix[] A_bak = new Matrix[A.Length];
            Matrix[] Z_bak = new Matrix[Z.Length];
            for (int l = 0; l < A.Length; l++)
            {
                A_bak[l] = A[l].Clone();
                if (l != 0)
                    Z_bak[l] = Z[l].Clone();
            }
            Select(Training, k);
            for (int l = 0; l < W.Length; l++)
            {
                var W_bak = W[l].Clone();
                //check Weights Gradients
                for (int row = 0; row < W[l].Rows; row++)
                {
                    if (!isWeightGradientCorrect)
                        break;
                    for (int col = 0; col < W[l].Cols; col++)
                    {
                        double lossAbove = 0;
                        double lossBelow = 0;
                        W[l][row, col] -= epsilonGradientCheck;
                        Forward();
                        Loss[i] = 0;
                        Criterion(i);
                        lossBelow = Loss[i];

                        W[l][row, col] += (2 * epsilonGradientCheck);
                        Forward();

                        Loss[i] = 0;
                        Criterion(i);
                        lossAbove = Loss[i];

                        var approxGrad = (lossAbove - lossBelow) / (2 * epsilonGradientCheck);
                        W[l] = W_bak;
                        if (Math.Abs(dW[l][row, col] - approxGrad) > gradientCheckThreshold)
                        {
                            isWeightGradientCorrect = false;
                            break;
                        }
                    }
                }
                //Check Bias Gradients
                for (int row = 0; row < b[l].Rows; row++)
                {
                    double lossAbove = 0;
                    double lossBelow = 0;
                    var b_bak = b[l].Clone();
                    b[l][row, 0] -= epsilonGradientCheck;
                    Forward();
                    Loss[i] = 0;
                    Criterion(i);
                    lossBelow = Loss[i];

                    b[l][row, 0] += (2 * epsilonGradientCheck);
                    Forward();
                    Loss[i] = 0;
                    Criterion(i);
                    lossAbove = Loss[i];

                    var approxGrad = (lossAbove - lossBelow) / (2 * epsilonGradientCheck);
                    b[l] = b_bak;
                    if (Math.Abs(db[l][row, 0] - approxGrad) > gradientCheckThreshold)
                    {
                        isBiasGradientCorrect = false;
                        break;
                    }
                }
            }
            Loss[i] = Loss_bak;
            return isWeightGradientCorrect && isBiasGradientCorrect;
        }
        static void InitNN(string[] args, StreamWriter writer)
        {
            var argsCounter = 1;
            writer.WriteLine("===============  Data             ===================");
            trainImagesFile = args[argsCounter++];
            writer.WriteLine("Training Image file: " + trainImagesFile);
            trainLabelsFile = args[argsCounter++];
            writer.WriteLine("Training Labels file: " + trainLabelsFile);
            testImagesFile = args[argsCounter++];
            writer.WriteLine("Testing Images file: " + testImagesFile);
            testLabelsFile = args[argsCounter++];
            writer.WriteLine("Testing Labels file: " + testLabelsFile);
            trainingSize = int.Parse(args[argsCounter++]);
            writer.WriteLine("Training Size: " + trainingSize);
            testingSize = int.Parse(args[argsCounter++]);
            writer.WriteLine("Testing Size: " + testingSize);
            writer.WriteLine("===============  Network Parameters ===================");
            batchSize = int.Parse(args[argsCounter++]);
            writer.WriteLine("Training Mini-Batch Size: " + batchSize);
            epochs = int.Parse(args[argsCounter++]);
            writer.WriteLine("Training Epochs: " + epochs);
            learningRate = double.Parse(args[argsCounter++]);
            writer.WriteLine("Learning Rate: " + learningRate);
            weightDecay = double.Parse(args[argsCounter++]);
            writer.WriteLine("Weight Decay: " + weightDecay);
            seed = int.Parse(args[argsCounter++]);
            writer.WriteLine("Random Seed: " + seed);
            problemType = args[argsCounter++];  //classification or regression
            writer.WriteLine("Problem Type: " + problemType);
            criterion = args[argsCounter++];  //mse or softmax
            writer.WriteLine("Cost Criterion: " + criterion);
            epsilonLeaky = double.Parse(args[argsCounter++]);  //threshold for Leaky ReLU
            writer.WriteLine("Leaky ReLU threshold: " + epsilonLeaky);
            epsilonGradientCheck = double.Parse(args[argsCounter++]);
            writer.WriteLine("Gradient Check Step: " + epsilonGradientCheck);
            gradientCheckThreshold = double.Parse(args[argsCounter++]);
            writer.WriteLine("Gradient Check Threshold: " + gradientCheckThreshold);
            L = int.Parse(args[argsCounter++]);
            layerFunctions = new string[L + 1];
            for (int l = 1; l <= L; l++)
            {
                layerFunctions[l] = args[argsCounter++];  //Sigmoid, Tanh, ReLU, or LReLU (Leaky ReLU) for now
                writer.WriteLine("Hidden Layer " + l + ": " + layerFunctions[l]);
            }

            writer.WriteLine("Number of Hidden Layers: " + L);
            layerSizes = new int[L + 2];
            W = new Matrix[L + 1];
            dW = new Matrix[L + 1];
            deltaW = new Matrix[L + 1];
            b = new Matrix[L + 1];
            db = new Matrix[L + 1];
            deltab = new Matrix[L + 1];
            errors = new Matrix[L + 2];
            Loss = new double[epochs];
            for (int i = 0; i <= L + 1; i++)
            {
                layerSizes[i] = int.Parse(args[argsCounter++]);
                writer.WriteLine("Size of Layer No. " + (i + 1) + ": " + layerSizes[i]);
            }
            Data = MNISTReader.ReadData(trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile);
            Training = Data.Item1;
            Testing = Data.Item1;
            acc_dW = new List<Matrix[]>();
            acc_db = new List<Matrix[]>();
            for (int i = 0; i <= L; i++)
            {
                var W_layer = new Matrix(layerSizes[i + 1], layerSizes[i]);
                var b_layer = new Matrix(layerSizes[i + 1], 1);
                var dW_layer = new Matrix(layerSizes[i + 1], layerSizes[i]);
                var db_layer = new Matrix(layerSizes[i + 1], 1);
                var deltaW_layer = new Matrix(layerSizes[i + 1], layerSizes[i]);
                var deltab_layer = new Matrix(layerSizes[i + 1], 1);
                var errors_layer = new Matrix(layerSizes[i + 1], 1);
                W_layer.Ramdomize(-0.01, 0.01);
                b_layer.Ramdomize(-0.01, 0.01);
                W[i] = W_layer;
                b[i] = b_layer;
                dW[i] = dW_layer;
                db[i] = db_layer;
                deltaW[i] = deltaW_layer;
                deltab[i] = deltab_layer;
                errors[i + 1] = errors_layer;
                for(int j=0; j <= batchSize; j++)
                {
                    acc_dW[j][i] = new Matrix(layerSizes[i + 1], layerSizes[i]);
                    acc_db[j][i] = new Matrix(layerSizes[i + 1], 1);
                }
            }
            A = new Matrix[L + 2];
            Z = new Matrix[L + 2];
            label = Matrix.CreateColumnMatrix(new double[layerSizes[L + 1]]);
        }
        static void Main(string[] args)
        {
            logFile = args[0];
            using (StreamWriter writer = new StreamWriter(logFile))
            {
                writer.AutoFlush = true;
                writer.WriteLine("===============  Reading Data & Initializing NN   ===================");
                InitNN(args, writer);
                writer.WriteLine("===============  NN Training                      ===================");

                var rnd = new Random(seed);
                TimeSpan ts;
                string elapsedTime;
                
                for (int i = 0; i < epochs; i++)
                {
                    Stopwatch epochStopWatch = new Stopwatch();
                    epochStopWatch.Start();
                    Training = Training.OrderBy(item => rnd.Next()).ToList<DigitImage>();
                    for (int j = 0; j < trainingSize; j += batchSize)
                    {
                        ZeroGrads();
                        for (int k = j; k < Math.Min(j + batchSize, Training.Count); k++)
                        {
                            Select(Training, k);
                            Forward();
                            Criterion(i);
                            Backward(k);
                        }
                        Accumulate();
                        Update();
                        ReportBatch(writer,i,j);
                    }
                    ReportPerf(writer, Training, i, "Training Performance");
                    ReportPerf(writer, Testing, i, "Testing Performance");
                    ReportLoss(writer, i);
                    ReportTime(writer, out ts, out elapsedTime, epochStopWatch);
                }
            }
        }
        private static void Select(List<DigitImage> data, int index)
        {
            A[0] = Matrix.CreateColumnMatrix(data[index].Flatten());
            label.Clear();
            label[Training[index].label, 0] = 1;
        }
        private static void ZeroGrads()
        {
            foreach (var delta in deltaW)
                delta.Clear();
            foreach (var delta in deltab)
                delta.Clear();
        }
        private static void Forward()
        {    
            for (int l = 0; l < L; l++)
            {
                Z[l + 1] = MatrixMath.Add(MatrixMath.Multiply(W[l], A[l]), b[l]);
                A[l + 1] = MatrixMath.F(Z[l + 1], layerFunctions[l + 1], epsilonLeaky, false); //false mean apply function, true means apply function derivative
            }
            Z[L + 1] = MatrixMath.Add(MatrixMath.Multiply(W[L], A[L]), b[L]); //last layer pre-activation
            if (criterion == "MSE")    //last layer activation 
                A[L + 1] = Z[L + 1].Clone();
            else if (criterion == "SoftMax")
            {
                A[L + 1] = new Matrix(10, 1);
                double Denom = MatrixMath.SumExp(Z[L + 1]);
                for (int c = 0; c < Z[L + 1].Rows; c++)
                    A[L + 1][c, 0] = Math.Exp(Z[L + 1][c, 0]) / Denom;
            }
        }
        private static void Criterion(int i)
        {
            if (criterion == "SoftMax")
            {
                for (int c = 0; c < layerSizes[L + 1]; c++)
                {
                    Loss[i] -= (label[c, 0] * Math.Log(A[L + 1][c, 0]));
                }
            }
            else if (criterion == "MSE")
            {
                for (int c = 0; c < layerSizes[L + 1]; c++)
                {
                    Loss[i] += 0.5 * Math.Pow((label[c, 0] - A[L + 1][c, 0]), 2);
                }
            }
        }
        private static void Backward(int k)
        {
            errors[L + 1] = MatrixMath.Subtract(A[L + 1], label); //error at output layer neurons
            for (int l = L; l >= 1; l--) // error at hidden layers neurons e[l] = W[l].Transpose * e[l+1]) . F'(Z[l])    where * is matrix multiplication and . is hadamard multiplication
            {
                errors[l] = MatrixMath.HadamardProduct(MatrixMath.Multiply(MatrixMath.Transpose(W[l]), errors[l + 1]), MatrixMath.F(Z[l], layerFunctions[l], epsilonLeaky, true));
            }
            for (int l = 0; l <= L; l++)
            {
                dW[l] = MatrixMath.Multiply(errors[l + 1], MatrixMath.Transpose(A[l]));
                db[l] = errors[l + 1];
            }

            for (int l = 0; l <= L; l++)
            {
                acc_dW[k][l] = dW[l].Clone();
                acc_db[k][l] = db[l].Clone();
                //deltaW[l] = MatrixMath.Add(deltaW[l], dW[l]);
                //deltab[l] = MatrixMath.Add(deltab[l], db[l]);
            }
        }
        private static void Accumulate()
        {
            for (int l = 0; l <= L; l++)
            {
                for(int k = 0; k < batchSize; k++)
                {
                    deltaW[l] = MatrixMath.Add(deltaW[l], acc_dW[k][l]);
                    deltab[l] = MatrixMath.Add(deltab[l], acc_db[k][l]);
                }
            }
        }
        private static void Update()
        {
            for (int l = 0; l <= L; l++)
            {
                deltaW[l] = MatrixMath.Multiply(deltaW[l], 1.0 / batchSize);
                deltab[l] = MatrixMath.Multiply(deltab[l], 1.0 / batchSize);
                W[l] = MatrixMath.Subtract(W[l], MatrixMath.Multiply(MatrixMath.Add(deltaW[l], MatrixMath.Multiply(W[l], weightDecay)), learningRate));
                b[l] = MatrixMath.Subtract(b[l], MatrixMath.Multiply(deltab[l], learningRate));
            }
        }
        private static void ReportLoss(StreamWriter writer, int i)
        {
            writer.WriteLine("=====>  Training Loss at Epoch " + i + ": " + (Loss[i] /= Training.Count));
            Console.WriteLine("=====>  Training Loss at Epoch " + i + ": " + (Loss[i] /= Training.Count));
        }

        private static void ReportTime(StreamWriter writer, out TimeSpan ts, out string elapsedTime, Stopwatch epochStopWatch)
        {
            epochStopWatch.Stop();
            ts = epochStopWatch.Elapsed;
            elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            writer.WriteLine("Training Time " + elapsedTime);
        }
        private static void ReportBatch(StreamWriter writer,int i,int j)
        {
            writer.WriteLine("=====>  Training Loss after Batch " + (j / batchSize + 1) + " on Epoch " + i + ": " + (Loss[i] /= Training.Count));
            for (int l = 0; l <= L; l++)
            {
                writer.WriteLine("\r\nEpoch " + (i + 1) + " Batch " + (j / batchSize + 1) + " Layer " + (l + 1) + " Weights Gradients Norm:" + MatrixMath.PNorm(deltaW[l], 2));
                writer.WriteLine("Epoch " + (i + 1) + " Batch " + (j / batchSize + 1) + " Layer " + (l + 1) + " Biases Gradients Norm:" + MatrixMath.PNorm(deltab[l], 2) + "\r\n");
            }
        }
        private static void ReportPerf(StreamWriter writer, List<DigitImage> data, int epoch, string whichPerformance)
        {
            var confusionMatrix = new Matrix(layerSizes[L + 1], layerSizes[L + 1]);
            double accuracy = 0;
            for (int k = 0; k < data.Count; k++)
            {
                var loss_bak = (double[])Loss.Clone();
                Select(data, k);
                Forward();
                var maxPred = MatrixMath.Max(A[L + 1]);
                var label = (int)data[k].label;
                confusionMatrix[label, maxPred.Item1]++;
            }
            accuracy = MatrixMath.Trace(confusionMatrix)/data.Count;
            writer.WriteLine(whichPerformance + " Performance at Epoch: " + epoch);
            writer.WriteLine(whichPerformance + " Confusion Matrix: ");
            writer.WriteLine(confusionMatrix);
            writer.WriteLine(whichPerformance + " Accuracy: " + accuracy);
        }
    }
}
