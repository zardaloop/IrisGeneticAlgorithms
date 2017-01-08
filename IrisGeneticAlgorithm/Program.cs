using HeatonResearchNeural.Feedforward;
using HeatonResearchNeural.Feedforward.Train.Genetic;
using IrisGeneticAlgorithm;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace IrisGeneticAlgorithm
{
    enum Labels{setosa,versicolor,virginica};

    class Program
    {
        public static double[][] input;

        public static double[][] label;

        private static string GetPath()
        {
            return Path.Combine(Environment.CurrentDirectory, @"dataset\irisdata.csv"); ;
        }

        private static double[][] GetInput()
        {
            string path = GetPath();
            StreamReader sr = new StreamReader(path);
            var lines = new List<double[]>();
            int Row = 0;
            while (!sr.EndOfStream)
            {
                string[] line = sr.ReadLine().Split(',');
                line = line.Take(line.Count() - 1).ToArray();    
                double[] doubleLine = Array.ConvertAll(line, Double.Parse);

                lines.Add(doubleLine);
                Row++;
                Console.WriteLine(Row);
            }
            
            return lines.ToArray();
        }

        private static double[][] GetLabel()
        {
            string path = GetPath();
            StreamReader sr = new StreamReader(path);
            var lines = new List<double[]>();
            int Row = 0;
            while (!sr.EndOfStream)
            {
                string[] line = sr.ReadLine().Split(',');
                line = line.Reverse().Take(1).ToArray();
                double[] doubleLine = line.Select(a => Convert.ToDouble(Enum.Parse(typeof(Labels), a))).ToArray();


                lines.Add(doubleLine);
                Row++;
                Console.WriteLine(Row);
            }

            return lines.ToArray();
        }

        static void Main(string[] args)
        {
            input = GetInput();
            label = GetLabel();

            FeedforwardNetwork network = new FeedforwardNetwork();
            network.AddLayer(new FeedforwardLayer(4));
            network.AddLayer(new FeedforwardLayer(5));
            network.AddLayer(new FeedforwardLayer(1));
            network.Reset();

            // train the neural network
            TrainingSetNeuralGeneticAlgorithm train = new TrainingSetNeuralGeneticAlgorithm(
                    network, false, input, label, 5000, 0.1, 0.25);

            int epoch = 1;

            do
            {
                train.Iteration();
                Console.WriteLine("Epoch #" + epoch + " Error:" + train.Error);
                epoch++;
            } while ((epoch < 5000) && (train.Error > 0.001));

            network = train.Network;

            // test the neural network
            Console.WriteLine("Neural Network Results:");
            for (int i = 0; i < label.Length; i++)
            {
                double[] actual = network.ComputeOutputs(input[i]);
                Console.WriteLine(input[i][0] + "," + input[i][1]
                        + ", actual=" + actual[0] + ",ideal=" + label[i][0]);
            }
        }
    }
}
