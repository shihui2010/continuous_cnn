import random
import numpy as np
from pydelay._dde23 import dde23


class BaseBatcher(object):
    def __init__(self, hps, tstart, tfinal, dt):
        self.input_length = hps["input_length"]
        self.output_length = hps["output_length"]
        self.input_channel = 1
        self.output_channel = 1

        uniform_sample = self.dde.sample(tstart=tstart, tfinal=tfinal, dt=dt)
        half = len(uniform_sample['x']) / 2
        xmax, xmin = np.max(uniform_sample['x']), np.min(uniform_sample['x'])
        self.x = (np.array(uniform_sample['x']) * 2 - (xmax + xmin)) / (xmax - xmin)
        self.train_x = [[i] for i in self.x[:half]]
        self.test_x = [[i] for i in self.x[half:]]
        self.t = np.linspace(0, (len(self.x) - 1)/3.0, len(self.x))

        self.train_t = self.t[:half]
        self.test_t = self.t[half:] - half/3.0

        self.offset_range = range(hps["offset_range"][0],
                                  hps["offset_range"][1])

    def next_train(self, batch_size):
        return self._next_batch(self.train_t, self.train_x, batch_size)

    def next_test(self, batch_size):
        return self._next_batch(self.test_t, self.test_x, batch_size)

    def _next_batch(self, tpool, xpool, batch_size):
        itvl_in, signals, itvl_out, labels = list(), list(), list(), list()
        for _ in range(batch_size):
            # uncomment for nonuniform sampling
            #offset = random.choice(self.offset_range)
            #tl = max(self.output_length + offset, self.input_length)
            #start = random.choice(range(len(self.train_t) - tl * 3))
            #idx = random.sample(range(start + 1, start + 3 * tl - 1), tl - 2)
            #idx.extend([start, start + 3 * tl - 1])
            #idx = sorted(idx)
            start = np.random.choice(np.arange(len(self._train_t) - self.output_length - self.input_length - self.offset[0]))
            idx = np.arange(start, start + self.input_length + self.output_length + self.offset[0])
            itvl_in.append([tpool[i] for i in idx[:self.input_length]])
            signals.append([xpool[i] for i in idx[:self.input_length]])
            itvl_out.append([tpool[idx[i]] for i in range(offset, len(idx))])
            labels.append([xpool[idx[i]] for i in range(offset, len(idx))])
        return itvl_in, signals, itvl_out, labels

    def long_pred(self, batch_size):
        idx = [311, 312, 317, 321, 322, 325, 331, 336, 338, 339, 341, 342,
               347, 348, 352, 356, 357, 358, 361, 364, 368, 369, 375, 380,
               386, 390, 395, 396, 397, 408, 409, 413, 416, 422, 423, 426,
               427, 428, 429, 430, 431, 432, 434, 439, 444, 445, 447, 455,
               457, 460, 466, 467, 468, 473, 476, 478, 480, 484, 485, 496,
               497, 498, 508, 509, 510, 512, 513, 514, 516, 521, 522, 525,
               528, 529, 532, 535, 536, 537, 539, 540, 546, 548, 552, 562,
               567, 570, 573, 577, 578, 585, 593, 596, 597, 599, 600, 602,
               603, 605, 607, 609]
        idx = idx[:self.input_length]
        itin, sig, itout, preds = list(), list(), list(), list()
        for bid in range(batch_size):
            itin.append([self.test_t[i] for i in idx])
            sig.append([self.test_x[i] for i in idx])
            itout.append([self.test_t[idx[-1] + bid]])
            preds.append([self.test_x[idx[-1] + bid]])
        return itin, sig, itout, preds

    def plot(self):
        try:
            import pylab as pl
            pl.plot(self.t, self.x)
            pl.xlabel("$t$")
            pl.show()
        except ImportError:
            print("Not support")


class MGlass(BaseBatcher):
    eqns = {'x': 'a * x(t-tau) / (1.0 + pow(x(t-tau),c)) - b*x'}

    #define the parameters
    params = {'tau': 17, 'a': 0.2, 'b': 0.1, 'c': 10}

    # Initialise the solver
    dde = dde23(eqns=eqns, params=params)
    dde.set_sim_params(tfinal=5000, dtmax=1.0)

    # set the history of to the constant function 0.5 (using a python lambda function)
    histfunc = {'x': lambda t: 1.2}
    dde.hist_from_funcs(histfunc, 51)
    dde.run()

    def __init__(self, hps):
        super(MGlass, self).__init__(hps, tstart=50, tfinal=5000, dt=2.0)


class Lorenz(BaseBatcher):
    eqns = {
        'x': 'sigma * (y - x)',
        'y': 'x * (r - z) - y',
        'z': 'x * y - b * z'
    }

    params = {
        'sigma': 10,
        'r': 28,
        'b': 8.0 / 3
    }

    dde = dde23(eqns=eqns, params=params)

    dde.set_sim_params(tfinal=50, dtmax=1)

    histfunc = {
        'x': lambda t: 0.1,
        'y': lambda t: 0,
        'z': lambda t: 0
    }
    dde.hist_from_funcs(histfunc)
    dde.run()

    def __init__(self, hps):
        super(Lorenz, self).__init__(hps, tstart=40, tfinal=50, dt=0.05)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # m = MGlass({"input_length": 100, "output_length": 12,
    #             "offset_range": [1, 5]})
    # itin, sig, itout, label = m.long_pred(20)

    l = Lorenz({"input_length": 13, "output_length": 1,
                "offset_range": [13, 16]})
    diffs = dict()
    for _ in xrange(10):
        itin, sig, itou, label = l.next_train(1)
        diff = itin[0][-1] - itin[0][0]
        print(diff)
        diffs[diff] = diffs.setdefault(diff, 0) + 1
    import matplotlib.pyplot as plt
    plt.plot(diffs.keys(), diffs.values(), "^b")
    plt.show()
