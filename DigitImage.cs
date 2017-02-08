using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace TestNN
{
    public class MNISTReader
    {
        //"t10k-labels.idx1-ubyte"  --> test labels file
        //"t10k-images.idx3-ubyte"  --> test images file
        //"train-labels-idx1-ubyte"  --> train labels file  
        //"train-images-idx3-ubyte"  --> train images file
        public static Tuple<List<DigitImage>, List<DigitImage>> ReadData(string trainImagesFile, string trainLabelsFile, string testImagesFile, string testLabelsFile)
        {
            var Training = new List<DigitImage>();
            var Testing = new List<DigitImage>();
            MNISTReader.ReadMNIST(trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile, Training, Testing);
            return new Tuple<List<DigitImage>, List<DigitImage>>(Training, Testing);
        }
        private static void ReadMNIST(string trainImagesFile, string trainLabelsFile, string testImagesFile, string testLabelsFile, List<DigitImage> Training, List<DigitImage> Testing)
        {
            try
            {
                Console.WriteLine("\nBegin Reading MNIST Data\n");
                FileStream ifsTrainImages =
                 new FileStream(trainImagesFile,
                 FileMode.Open); // train images

                FileStream ifsTrainLabels =
                new FileStream(trainLabelsFile,
                FileMode.Open); // train labels

                FileStream ifsTestImages =
                new FileStream(testImagesFile,
                FileMode.Open); // test images

                FileStream ifsTestLabels =
                 new FileStream(testLabelsFile,
                 FileMode.Open); // test labels

                BinaryReader brTrainImages =
                 new BinaryReader(ifsTrainImages);
                BinaryReader brTrainLabels =
                 new BinaryReader(ifsTrainLabels);
                BinaryReader brTestImages =
                new BinaryReader(ifsTestImages);
                BinaryReader brTestLabels =
                 new BinaryReader(ifsTestLabels);


                int magic1 = brTrainImages.ReadInt32(); // discard
                int numImages = brTrainImages.ReadInt32();
                int numRows = brTrainImages.ReadInt32();
                int numCols = brTrainImages.ReadInt32();
                int magic2 = brTrainLabels.ReadInt32();
                int numLabels = brTrainLabels.ReadInt32();

                magic1 = brTestImages.ReadInt32(); // discard
                numImages = brTestImages.ReadInt32();
                numRows = brTestImages.ReadInt32();
                numCols = brTestImages.ReadInt32();
                magic2 = brTestLabels.ReadInt32();
                numLabels = brTestLabels.ReadInt32();
                // each train image
                for (int di = 0; di < 60000; ++di)
                {
                    ReadImage(Training, brTrainImages, brTrainLabels);
                } // each image
                // each test image
                for (int di = 0; di < 10000; ++di)
                {
                    ReadImage(Testing, brTestImages, brTestLabels);
                } // each image

                ifsTestImages.Close();
                brTestImages.Close();
                ifsTestLabels.Close();
                brTestLabels.Close();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Environment.Exit(0);
            }
            Console.WriteLine("\nDone Reading MNIST Data\n");
        }

        private static void ReadImage(List<DigitImage> Data, BinaryReader brImages, BinaryReader brLabes)
        {
            byte[,] pixels = new byte[28,28];
            
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    byte b = brImages.ReadByte();
                    pixels[i,j] = b;
                }
            }

            byte lbl = brLabes.ReadByte();

            DigitImage dImage =
              new DigitImage(pixels, lbl);
            Data.Add(dImage);
        }
    }
    public class DigitImage
    {
        public byte[,] pixels;
        public byte label;

        public DigitImage(byte[,] pixels,
          byte label)
        {
            this.pixels = new byte[28,28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i,j] = pixels[i,j];

            this.label = label;
        }
        public double[,] ToDouble()
        {
            var rows = pixels.GetUpperBound(0) + 1;
            var columns = pixels.GetUpperBound(0) + 1;
            var result = new double[rows , columns];

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    result[r,c] = (double)pixels[r, c];
                }
            }
            return result;
        }
        public byte[] Flatten()
        {
            var rows = pixels.GetUpperBound(0) + 1;
            var columns = pixels.GetUpperBound(0) + 1;
            var result = new byte[rows * columns];

            int index = 0;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    result[index++] = pixels[r, c];
                }
            }
            return result;
        }
        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i,j] == 0)
                        s += " "; // white
                    else if (this.pixels[i,j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } 
    }
}
