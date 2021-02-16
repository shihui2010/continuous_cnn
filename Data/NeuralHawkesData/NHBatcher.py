import random
import pickle
import os
import numpy as np


class PointProcessBatcher(object):
    def __init__(self, hps):
        assert hps["output_length"] == 1, "compatibility check"
        assert hps["input_length"] < 50, "compatibility check"

        self.dataset = hps["dataset"]
        _path = os.path.join(os.path.dirname(__file__), hps["dataset"])
        with open(os.path.join(_path, "train.pkl")) as fp:
            train = pickle.load(fp)["train"]
        with open(os.path.join(_path, "test.pkl")) as fp:
            test = pickle.load(fp)["test"]
        with open(os.path.join(_path, "dev.pkl")) as fp:
            dev = pickle.load(fp)["dev"]

        self.input_channel = hps["input_channel"]
        self.target_class = range(self.input_channel)
        self.output_channel = self.input_channel
        self.input_channel = self.output_channel
        self.output_length = 1
        self.input_length = hps["input_length"]

        norm_factor = None
        train_itvl, train_sig, norm_factor = self.transform_data(train, norm_factor=norm_factor)
        self.train_size = train_itvl.shape[0]
        test_itvl, test_sig, _ = self.transform_data(test, norm_factor=norm_factor)
        self.test_size = test_itvl.shape[0]
        dev_itvl, dev_sig, _ = self.transform_data(dev, norm_factor=norm_factor)
        self.dev_size = dev_itvl.shape[0]
        
        self.norm_factor = norm_factor
        print "-" * 15, "Data Load Info", "-" * 15

        self._t = np.concatenate([train_itvl, test_itvl, dev_itvl], axis=0)
        self._v = np.concatenate([train_sig, test_sig, dev_sig], axis=0)
        print "Loading from", self.dataset
        print "Ovarall average interval", np.mean(self._t)

        reduce = np.add.reduce
        train_unbalance = reduce(train_sig[:,-1], axis=0).astype(float)
        train_unbalance /= float(train_sig.shape[0])
        test_unbalance = reduce(test_sig[:,-1], axis=0).astype(float)
        test_unbalance /= float(test_sig.shape[0])
        dev_unbalance = reduce(dev_sig[:,-1], axis=0).astype(float)
        dev_unbalance /= float(dev_sig.shape[0])
        
        print "train label balance", train_unbalance, "train size", self.train_size
        print "test label balance", test_unbalance, "test size", self.test_size
        print "dev label balance", dev_unbalance, "dev size", self.dev_size
        pool = range(self.train_size + self.test_size + self.dev_size)

        self.train_pool = pool[:self.train_size]
        self.test_pool = pool[self.train_size: self.train_size + self.test_size]
        self.dev_pool = pool[-self.dev_size:]
        random.shuffle(self.train_pool)
        self.train_idx = 0
        self.test_idx = 0
        self.dev_idx = 0

    def transform_data(self, data_set, norm_factor=None):
        itvl = list()
        signal = list()
        sam_idx = 0
        for item in data_set:
            offset = 0
            while offset < len(item) - self.input_length - 1:
                itvl.append(np.zeros([self.input_length + 1]))
                signal.append(
                    np.zeros([self.input_length + 1, self.input_channel],
                             dtype=np.int8))
                for seq_idx, seq in enumerate(
                        item[offset: offset + self.input_length + 1]):
                    itvl[sam_idx][seq_idx] = seq["time_since_last_event"]
                    # time_since_last_event, time_since_start
                    signal[sam_idx][seq_idx][seq["type_event"]] = 1
                offset += 1 #self.input_length + 1
                sam_idx += 1
        itvl = np.stack(itvl, axis=0)
        signal = np.stack(signal, axis=0)
        print "Average Interval", np.mean(itvl)
        print "Median of Interval", np.median(itvl)
        print "Historgram", np.histogram(itvl)
        if norm_factor is None:
            if "bookorder" in self.dataset:
                norm_factor = np.median(itvl)
            elif "so" in self.dataset:
                norm_factor = np.mean(itvl)
            else:
                norm_factor = 1
        print "Norm factor is", norm_factor
        itvl /= norm_factor
        print "Normalized Average Interval", np.mean(itvl)
        print "-" * 50
        # interval to timestamp
        for sam_idx in range(itvl.shape[0]):
            for seq_idx in range(itvl.shape[1] - 1):
                itvl[sam_idx][seq_idx + 1] += itvl[sam_idx][seq_idx]
        if self.target_class is None:
            return itvl, signal, norm_factor
        return itvl, signal[:,:,self.target_class], norm_factor

    def next_train(self, batch_size):
        if self.train_idx + batch_size >= self.train_size:
            self.train_idx = 0
            random.shuffle(self.train_pool)
            self.train_size = len(self.train_pool)
        itvl_in, sig, itvl_out, labels = self._next_batch(
            self.train_idx, self.train_pool, batch_size)
        self.train_idx += batch_size
        return itvl_in, sig, itvl_out, labels

    def next_test(self, batch_size):
        if self.test_idx + batch_size >= self.test_size:
            self.test_idx = 0
            self.test_size = len(self.test_pool)
        itvl_in, sig, itvl_out, labels = self._next_batch(
            self.test_idx, self.test_pool, batch_size)
        self.test_idx += batch_size
        return itvl_in, sig, itvl_out, labels

    def _next_batch(self, start_idx, idx_pool, batch_size):
        indices = idx_pool[start_idx: start_idx + batch_size]
        itvl_in, sig, itvl_out, labels = list(), list(), list(), list()
        for idx in indices:
            # if self._t[idx][-1] - self._t[idx][-2] > 4:
            #    idx_pool.remove(idx)
            itvl_in.append(self._t[idx][:-1])
            sig.append(self._v[idx][:-1])
            itvl_out.append([self._t[idx][-1]])
            labels.append([self._v[idx][-1]])
        
        # repeat sample to match batch size
        while len(itvl_in) < batch_size:
            itvl_in.append(itvl_in[-1])
            sig.append(sig[-1])
            itvl_out.append(itvl_out[-1])
            labels.append(labels[-1])
        # if len(self.target_class) == 2:
            # print [[l[0][:1]] for l in labels]
            # return itvl_in, sig, itvl_out, [[l[0][:1]] for l in labels]
        return itvl_in, sig, itvl_out, labels

    def iter_test(self, batch_size):
        idx = 0
        while idx < len(self.test_pool) - batch_size:
            yield self._next_batch(idx, self.test_pool, batch_size)
            idx += batch_size
    
    def iter_dev(self, batch_size):
        idx = 0
        while idx < len(self.dev_pool) - batch_size:
            yield self._next_batch(idx, self.dev_pool, batch_size)
            idx += batch_size

    def iter_train(self, batch_size):
        idx = 0
        while idx < len(self.train_pool) - batch_size:
            yield self._next_batch(idx, self.train_pool, batch_size)
            idx += batch_size
        np.random.shuffle(self.train_pool) 


if __name__ == "__main__":
    batcher = PointProcessBatcher({"dataset": "data_bookorder/fold1",
                               "input_length": 20, "output_length": 1,
                               "num_labels": 3, "target_class": [1]})

    for _ in range(10):
        itvl_in, sig, itvl_out, label = batcher.next_train(20)
