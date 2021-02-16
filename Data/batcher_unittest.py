import numpy as np
import unittest
from unittest import TestCase
from sinusoid_signal import SignalGen as SingleSine
from sinusoid_signal import DoubleSignal
from sinusoid_signal import PredSignal
from mnist.batcher import Batcher as MnistBatcher
from UCR.UCRBatcher import Batcher as UCRBatcher
from data_market.DMBatcher import Batcher as DMBatcher
from UCI.Batcher import Batcher as UCIBatcher


class TestAllBatchers(TestCase):
    def setUp(self):
        batchers = list()
        names = list()

        batchers.append(SingleSine({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": True,
                                    "seq_level": True}))
        batchers.append(SingleSine({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": True,
                                    "seq_level": False}))
        batchers.append(SingleSine({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": False,
                                    "seq_level": True}))
        batchers.append(SingleSine({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": False,
                                    "seq_level": False}))
        names.extend(
            ["SingleSine_Uniform_Seq", "SingleSine_Uniform_Sample",
            "SingleSine_Nonuniform_Seq", "SingleSine_Nonuniform_Sample"])

        batchers.append(DoubleSignal({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": True,
                                    "seq_level": True}))
        batchers.append(DoubleSignal({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": True,
                                    "seq_level": True}))
        batchers.append(DoubleSignal({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": True,
                                    "seq_level": True}))
        batchers.append(DoubleSignal({"input_length": 1000,
                                    "interval": 2.0,
                                    "uniform": True,
                                    "seq_level": True}))
        names.extend(
            ["DoubleSine_Uniform_Seq", "DoubleSine_Uniform_Sample",
            "DoubleSine_Nonuniform_Seq", "Doubleine_Nonuniform_Sample"])

        batchers.append(PredSignal({"input_length": 100,
                                    "interval": 1.0,
                                    "uniform": True,
                                    "freq_range": [2, 100],
                                    "delta_t": 10}))
        names.extend("SinePrediction")

        batchers.append(MnistBatcher("mnist/1024_1d",
                                     {"onehot": True}))
        batchers.append(MnistBatcher("mnist/1024_1d",
                                     {"onehot": False}))
        names.extend(["MNIST_1D_OH", "MNIST_1D"])

        batchers.append(UCRBatcher("StarLightCurves", input_length=10,
                    seq_level=True, prediction=True))
        batchers.append(UCRBatcher("50words", input_length=10,
                    seq_level=True, prediction=False))
        batchers.append(UCRBatcher("ScreenType", input_length=10,
                    seq_level=False, prediction=False))
        names.extend(["UCR_SLC", "UCR_50W", "UCR_ST"])

        batchers.append(DMBatcher("5.csv", 16))
        names.append("DMBatcher")

        batchers.append(UCIBatcher("gesture", 25, True, True))
        batchers.append(UCIBatcher("gesture", 5, False, True))
        batchers.append(UCIBatcher("gesture", 40, False, False))
        names.extend(["UCI_Regression", "UCI_Seq_Class", "UCI_Sam_Class"])

        self.batchers = batchers
        self.names = names

    def test_all(self):
        for b, n in zip(self.batchers, self.names):
            self.assertHasProperAttributes(b, n)
            self.assertShapeCorrect(b, n)

    def assertHasProperAttributes(self, batcher, name):
        attributes = ["input_length", "output_length", "input_channel",
                      "output_channel"]
        for att in attributes:
            self.assertTrue(hasattr(batcher, att),
                            name + " has no property " + att)

    def assertShapeCorrect(self, batcher, name):
        """
        interval should be 2D, with shape [batch_size, input_length]
        signal should be 3d, with shape [batch_size, input_length, input_channel]
        label should be 3d, wish shape [batch_size, output_length, out_channel]
        """
        batch_size = int(np.random.rand() * 100) + 1
        for method in [batcher.next_train, batcher.next_test]:
            interval, signal, label = method(batch_size)

            interval = np.array(interval)
            shape = (batch_size, batcher.input_length)
            self.assertTupleEqual(
                interval.shape, shape,
                name + " Interval Dimension Error: " + str(interval.shape)
                + " vs. " + str(shape))

            signal = np.array(signal)
            shape = (batch_size, batcher.input_length, batcher.input_channel)
            self.assertTupleEqual(
                signal.shape, shape,
                name + " Signal Dimension Error: " + str(signal.shape)
                + " vs. " + str(shape))

            label = np.array(label)
            shape = (batch_size, batcher.output_length, batcher.output_channel)
            self.assertTupleEqual(
                label.shape, shape,
                name + " Label Dimension Error: " + str(label.shape)
                + " vs. " + str(shape))


if __name__ == "__main__":
    unittest.main()