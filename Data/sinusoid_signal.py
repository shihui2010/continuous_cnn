import random
import math


def truncated_possion(resolution, confidence=0.9):
    interval = random.expovariate(resolution)
    return interval


def asignal(intervals, periods):
    return [[sum(math.sin(itv * 2 * math.pi / perd) / len(periods)
                 for perd in periods)] for itv in intervals]


def aninterval(resolution, uniform, seq_length):
    random_start = random.uniform(0, resolution)
    if not uniform:
        intervals = [truncated_possion(resolution) + random_start]
        for index in xrange(1, seq_length):
            intervals.append(
                truncated_possion(resolution) + intervals[index - 1])
    else:
        intervals = [resolution * i + random_start
                     for i in range(seq_length)]
    return intervals


class PredSignal(object):
    def __init__(self, hps):
        self.freq_range = hps["freq_range"]

        self.input_channel = 1
        self.input_length = hps["input_length"]
        self.output_channel = 1
        self.output_length = hps["output_length"]

        self.resolution = hps["interval"]
        self.uniform = hps["uniform"]
        self.offset_range = range(hps["offset_range"][0],
                                  hps["offset_range"][1])

    def _next_batch(self, batch_size, offset):
        signals, preds = list(), list()
        intervals_out = list()
        intervals_in = list()
        total_len = max(self.input_length, self.output_length + offset)
        for _ in range(batch_size):
            freq = random.choice(range(self.freq_range[0], self.freq_range[1]))

            itvl = aninterval(self.resolution, self.uniform, total_len)
            signal = asignal(itvl, [freq])
            signals.append(signal[:self.input_length])
            preds.append(signal[offset: offset + self.output_length])

            intervals_in.append(itvl[:self.input_length])
            intervals_out.append(itvl[offset: offset + self.output_length])

        return intervals_in, signals, intervals_out, preds

    def next_train(self, batch_size):
        return self._next_batch(batch_size, random.choice(self.offset_range))

    def next_test(self, batch_size):
        return self._next_batch(batch_size, random.choice(self.offset_range))
