from __future__ import division
import tensorflow as tf
from model import Model, TwoHot
from loss_funcs import softmax_cross_entropy_with_logits_loss


class SimpleRNN(Model):
    def __init__(self, hps, verbose=False, logdir=None,
                 loss_func=softmax_cross_entropy_with_logits_loss):
        self._hps = hps
        self.restore = hps["restore"]
        self.learning_rate = hps["learning_rate"]
        self.verbose = verbose

        self.twohots = TwoHot(
            hps["twohots"], hps["truncate_length"], hps["delta"])
        super(SimpleRNN, self).__init__(hps=hps, logdir=logdir,
                                         loss_func=loss_func)

    def _add_placeholder(self):
        # inputs
        self.x = tf.placeholder(
            tf.float32,
            [None, self._hps["input_length"], self._hps["input_channels"]])
        self.t = tf.placeholder(
            tf.float32,
            [None, self._hps["input_length"], self.twohots.thvec_len])

        # labels
        self._target = tf.placeholder(tf.float32,
            [None, self._hps["output_length"], self._hps["num_labels"]])
        # length of the samples -> for static rnn
        self.lens = tf.ones(shape=[self._hps["batch_size"]])
        self.lens = tf.cast(self._hps["input_length"] * self.lens,
                            dtype=tf.int32)

    def _build_graph(self):
        if self._hps["embed_signal"]:
            x = self._embed_signal(self.x)
        else:
            x = self.x

        self.input = tf.unstack(tf.concat([x, self.t], axis=-1), axis=1)

        rnn_output = self._add_rnn()
        if self.loss_func.__name__ == "pp_likelihood":
            self._output = self.add_customized_layers(
                rnn_output, [{"type": "pp_output", "act_fn": None}])
        else:
            # weights from input to hidden
            weights = tf.Variable(tf.random_normal(
                [self._hps["num_filters"], self._hps["num_labels"]],
                dtype=tf.float32), name="Weights_0")

            biases = tf.Variable(
                tf.random_normal([self._hps["num_labels"]], dtype=tf.float32),
                name="Bias_0")
            output = tf.nn.bias_add(tf.matmul(rnn_output, weights), biases)
            self._output = tf.expand_dims(output, axis=1)

    def _add_rnn(self):
        cells = tf.nn.rnn_cell.BasicRNNCell(self._hps["num_filters"])
        states, output = tf.nn.static_rnn(cells, self.input, dtype=tf.float32)

        return output

    def _transform_itvl(self, intervals):
        """
        transform time series to time intervals
        """
        target_interval = list()
        for interval in intervals:
            if len(interval) == self._hps["input_length"]:
                if self.twohots.twohots:
                    this_interval = [self.twohots.scalar2vec(interval[0])]
                else:
                    this_interval = [[interval[0]]]
            elif len(interval) == self._hps["input_length"] + 1:
                this_interval = []
            else:
                raise IndexError("interval length %s does not "
                                 "match input length" % (len(interval)))

            for index in range(1, len(interval)):
                if self.twohots.twohots:
                    this_interval.append(
                        self.twohots.scalar2vec(
                            interval[index] - interval[index - 1]))
                else:
                    this_interval.append(
                        [interval[index] - interval[index - 1]])
            target_interval.append(this_interval)
        return target_interval

    def _run(self, sess, signal, interval, target, *optional, **kwargs):
        interval = self._transform_itvl(interval)
        feed_dict = {self.x: signal, self.t: interval, self._target: target}
        self._update_dict(feed_dict, **kwargs)
        options = sess.run(optional, feed_dict=feed_dict)
        return options

    def train(self, sess, interval, signal, target):
        gs, acc, loss, _, smy = self._run(
            sess, signal, interval, target,
            self.global_step, self._accuracy, self._loss,
            self._train, self._merged_summary_op)
        self._summary_writer.add_summary(smy, global_step=gs)
        if self.train_epoches % 1000 == 0:
            self.saver.save(sess, self.log_dir + "/model.ckpt", global_step=gs)
        return gs, acc, loss

    def evaluate(self, sess, interval, signal, target):
        gs, acc, loss = self._run(
            sess, signal, interval, target,
            self.global_step, self._accuracy, self._loss)
        return gs, acc, loss
