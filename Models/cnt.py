import tensorflow as tf
from model import Model, TwoHot
from loss_funcs import softmax_cross_entropy_with_logits_loss


class CNT(Model):
    def __init__(self, hps, verbose=False, logdir=None,
                 loss_func=softmax_cross_entropy_with_logits_loss):
        self.restore = False
        self.learning_rate = hps["learning_rate"]
        self.verbose = verbose
        self.debug = False
        self.twohots = TwoHot(
            hps["twohots"], hps["truncate_length"], hps["delta"])
        super(CNT, self).__init__(hps, logdir=logdir, loss_func=loss_func)

    def _add_placeholder(self):
        self._signal_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._hps['input_length'],
                   self._hps["input_channels"]],
            name="Signals")
        tf.add_to_collection("Signals", self._signal_input)

        self._target = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._hps['output_length'],
                   self._hps['num_labels']],
            name="Targets"
        )
        tf.add_to_collection("Targets", self._target)

        self._interval_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._hps["input_length"], self.twohots.thvec_len],
            name="Intervals")
        tf.add_to_collection("interval_input", self._interval_input)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.add_to_collection("global_step", self.global_step)

    def _interval_transform(self, intervals):
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

    def _add_cnt_layer(self):
        cnt_config = self._hps["cnt_config"]
        pad = cnt_config.setdefault("padding", "SAME")

        # conv signal
        if self._hps["embed_signal"]:
            signal = self._embed_signal(self._signal_input)
        else:
            signal = self._signal_input

        s_kernel_shape = [cnt_config["signal_kernel"],
                          signal.get_shape().as_list()[-1],
                          cnt_config["channels"]]

        hidden_filter = tf.Variable(
            tf.truncated_normal(s_kernel_shape, stddev=0.01),
            name="phi_s")

        hidden_conved = tf.nn.conv1d(
            signal, hidden_filter, stride=1, padding=pad,
            name="Convolution_s")

        # conv time interval
        t_kernel_shape = [cnt_config["itvl_kernel"],
                          self.twohots.thvec_len,
                          cnt_config["channels"]]
        time_filter = tf.Variable(
            tf.truncated_normal(t_kernel_shape, stddev=0.01),
            name="phi_t")
        time_conved = tf.nn.conv1d(
            self._interval_input, time_filter, stride=1, padding=pad,
            name="Convolution_t")
        last_state = tf.nn.relu(tf.add(time_conved, hidden_conved))
        return last_state

    def _build_graph(self):
        if self.debug:
            self.hidden_conveds = list()
            self.hidden_act_conveds = list()
            self.filters = list()

        last_state = self._add_cnt_layer()
        if self.verbose:
            print last_state
        self._output = self.add_customized_layers(
            last_state, self._hps["layers"])

    def train(self, sess, signal, interval, target):
        acc, loss, gs, _, smy = self._run(
            sess, signal, interval, target,
            self._accuracy, self._loss, self.global_step, self._train,
            self._merged_summary_op)
        self._summary_writer.add_summary(smy, global_step=gs)
        self.train_epoches += 1
        if self.train_epoches % 500 == 0:
            self.saver.save(sess, self.log_dir + "/model.ckpt", global_step=gs)
        return gs, acc, loss

    def evaluate(self, sess, signal, interval, target):
        return self._run(sess, signal, interval, target,
                         self.global_step, self._accuracy, self._loss)

    def _run(self, sess, signal, interval, target, *optional, **kwargs):
        tsf_interval = self._interval_transform(interval)
        feed_dict = {self._signal_input: signal,
                     self._interval_input: tsf_interval,
                     self._target: target}
        self._update_dict(feed_dict, **kwargs)
        return sess.run(optional, feed_dict=feed_dict)
