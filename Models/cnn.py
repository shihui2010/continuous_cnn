import tensorflow as tf
from model import Model
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from loss_funcs import softmax_cross_entropy_with_logits_loss


class BaseCNN(Model):
    """
    input should be 1D, with N attributes specified by hps["input_channels"]
    output should be 1D, the length of which should agree with graph structure
    """
    def __init__(
            self, hps, verbose=False, logdir=None,
            loss_func=softmax_cross_entropy_with_logits_loss):
        self.restore = False
        self.learning_rate = hps["learning_rate"]
        self.verbose = verbose
        self.record_variable = False
        self.model_name = "CNN"
        super(BaseCNN, self).__init__(hps, logdir=logdir,
                                  loss_func=loss_func)

    def _add_placeholder(self):
        self._signal_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._hps["input_length"],
                   self._hps["input_channels"]], name="input")
        tf.add_to_collection("signal_input", self._signal_input)

        self._target = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._hps["output_length"], self._hps["num_labels"]],
            name="Target")
        tf.add_to_collection("target", self._target)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.add_to_collection("global_step", self.global_step)

    def _build_graph(self):
        if self._hps["embed_signal"]:
            signal = self._embed_signal(self._signal_input)
        else:
            signal = self._signal_input
        self._output = self.add_customized_layers(
            signal, self._hps["layers"])


class CNN(BaseCNN):

    def train(self, sess, interval, signal, target, *option, **unused):
        acc, loss, gs, _, smy, opt = self._run(
            sess, signal, interval, target,
            self._accuracy,
            self._loss,
            self.global_step,
            self._train,
            self._merged_summary_op,
            option
        )
        self._summary_writer.add_summary(smy, global_step=gs)
        self.train_epoches += 1
        if self.train_epoches % 500 == 0:
            self.saver.save(sess,
                            self.log_dir + "/" + self.model_name + ".ckpt",
                            global_step=self.global_step)
        if len(option) < 1:
            return gs, acc, loss
        return gs, acc, loss, opt

    def evaluate(self, sess, interval, signal, target, *option, **unused):
        acc, gs, loss, opt = self._run(
            sess, signal, interval, target,
            self._accuracy,
            self.global_step,
            self._loss,
            option
        )
        if len(option) < 1:
            return gs, acc, loss
        return gs, acc, loss, opt

    def _run(self, sess, signal, interval, target, *optional, **kwargs):
        feed_dict = {
                self._signal_input: signal,
                self._target: target}
        self._update_dict(feed_dict, **kwargs)
        result = sess.run(optional,
                          feed_dict=feed_dict)
        return result


class _pwc_interp(object):
    def __init__(self, timestamp, signal):
        self.timestamp = timestamp
        self.signal = signal

    def __call__(self, vec_t):
        return [self.__interp(x) for x in vec_t]

    def __interp(self, x):
        if x <= self.timestamp[0]:
            return self.signal[0]
        if x >= self.timestamp[-1]:
            return self.signal[-1]
        start, end = 0, len(self.timestamp)
        while start + 1 < end:
            mid = int((start + end) / 2)
            if self.timestamp[mid] > x:
                end = mid - 1
            elif self.timestamp[mid] == x:
                return self.signal[mid]
            elif mid == start:
                break
            else:
                start = mid
        return self.signal[start]


class _sinc_interp(object):
    def __init__(self, timestamp, signal):
        self.timestamp = timestamp
        self.signal = signal

    def __call__(self, vec_t):
        return [self.__interp(x) for x in vec_t]

    def __interp(self, x):
        sum = 0
        for t, s in zip(self.timestamp, self.signal):
            sum += s * self.__sinc(x - t)
        return sum

    def __sinc(self, x, a=1.0):
        if x == 0:
            return 1
        return a * np.sin(np.pi * x / a) / (np.pi * x)


class ICNN(BaseCNN):
    def __init__(self, hps, verbose=False, logdir=None,
                 loss_func=softmax_cross_entropy_with_logits_loss):
        print "ICNN with kernel:", hps["kernel"]
        super(ICNN, self).__init__(hps, verbose=verbose, logdir=logdir,
                                   loss_func=loss_func)

    def _interp(self, interval, signal):
        """
        :param interval: a single interval, with length: input_length (+ 1)
        :param signal: a single signal, with length: input_length
        :return:
        """
        if len(interval) == len(signal):
            new_t = [interval[0] + idx for idx in range(len(signal))]
        else:
            new_t = [interval[-1] - 1 - idx
                     for idx in range(len(signal), 0, -1)]
            interval = interval[:-1]
        for i in range(1, len(interval)):
            if interval[i] == interval[i - 1]:
                interval[i] += 1e-5
            if interval[i] < interval[i - 1]:
                interval[i] = interval[i - 1] + 1e-5
 
        # for each channel
        new_signal = np.zeros_like(signal)
        for c_id in range(len(signal[0])):
            sig_c = [i[c_id] for i in signal]            

            if self._hps["kernel"] == "linear":
                f = interp1d(interval, sig_c, kind="linear")
            elif self._hps["kernel"] == "quadratic":
                f = interp1d(interval, sig_c, kind="quadratic")
            elif self._hps["kernel"] == "cubic":
                f = CubicSpline(interval, sig_c)
            elif self._hps["kernel"] == "constant":
                f = _pwc_interp(interval, sig_c)
            elif self._hps["kernel"] == "sinc":
                f = _sinc_interp(interval, sig_c)
            else:
                raise ValueError(self._hps["kernel"] + " is not supported")
            for t_id, i in enumerate(new_t):
                if i < interval[0]:
                    new_signal[t_id, c_id] = sig_c[0]
                elif i > interval[-1]:
                    new_signal[t_id, c_id] = sig_c[-1]
                else:
                    new_signal[t_id, c_id] = f([i])[0]
        return new_signal

    def train(self, sess, interval, signal, target, *option, **kwargs):
        acc, loss, gs, _, smy, opt = self._run(
            sess, signal, interval, target,
            self._accuracy,
            self._loss,
            self.global_step,
            self._train,
            self._merged_summary_op,
            option
        )
        if acc > 0:
            self._summary_writer.add_summary(smy, global_step=gs)
            self.train_epoches += 1
            if self.train_epoches % 500 == 0:
                self.saver.save(sess,
                                self.log_dir + "/" + self.model_name + ".ckpt",
                                global_step=self.global_step)
        if len(option) < 1:
            return gs, acc, loss
        return gs, acc, loss, opt

    def evaluate(self, sess, interval, signal, target, *option, **kwargs):
        acc, loss, gs, opt = self._run(
            sess, signal, interval, target,
            self._accuracy,
            self._loss,
            self.global_step,
            option
        )
        if len(option) < 1:
            return gs, acc, loss
        return gs, acc, loss, opt

    def _run(self, sess, signal, interval, target, *optional, **kwargs):
        signal = self.interpolate(interval, signal)
        feed_dict = {
                self._signal_input: signal,
                self._target: target}
        self._update_dict(feed_dict, **kwargs)
        result = sess.run(optional,
                          feed_dict=feed_dict)
        return result

    def interpolate(self, intervals, signals):
        uni_signal = [self._interp(it, si)
                      for it, si in zip(intervals, signals)]
        return uni_signal


if __name__ == "__main__":
    # test piecewise constant interpolation

    t = [2, 3, 8, 9, 12, 34]
    signal = [-1, 1, -3, 5, 7, -9]
    f = _pwc_interp(t, signal)
    output = f([1, 2.5, 7, 8, 10, 25, 35, 40])
    target = [-1, -1, 1, 1, 5, 7, -9, -9]
    for o, t in zip(output, target):
        assert o == t, "piecewise constant interp error"

    # test sinc interpolation
    t = range(10)
    signal = [np.sin(x) for x in t]
    f = _sinc_interp(t, signal)
    t_new = [i + 0.5 for i in range(10)]
    target = [1.2, 3.3, 1.8, -1.0, -3.1, -2.1, 0.6, 3.0, 2.4, 0.4]
    for o, t in zip(output, target):
        assert abs(o - t) > 0.1, "sinc interp error"
