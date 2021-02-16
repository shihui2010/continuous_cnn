from time import gmtime, strftime
import json
import re
import numpy as np
import os
import tensorflow as tf
from pp_layers import pp_output_layer, pp_embed_signal
from loss_funcs import softmax_cross_entropy_with_logits_loss


class Model(object):
    def __init__(
            self, hps, logdir=None,
            loss_func=softmax_cross_entropy_with_logits_loss,
            hist=False):
        print ""
        print "-" * 14, "Model Info", "-" * 14
        self._hps = self._check_hps(hps)
        self.hist = hist
        self.loss_func = loss_func
        if self.restore:
            self.global_step = tf.get_collection("global_step")[0]
        else:
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            tf.add_to_collection("global_step", self.global_step)
        self.verbose = True
        self.train_epoches = 0

        # for direct time prediction
        if loss_func.__name__ == "weighted_l2":
            if self.restore:
                self._mask = tf.get_collection("Mask")[0]
            else:
                self._mask = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self._hps['output_length'],
                           self._hps['num_labels']],
                    name="Mask")
                tf.add_to_collection("Mask", self._mask)

        elif loss_func.__name__ == "pp_likelihood":
            if self.restore:
                self._dt = tf.get_collection("dt")[0]
            else:
                self._dt = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self._hps["output_length"]],
                    name="dt")
                # output_length should be one
                tf.add_to_collection("dt", self._dt)
            assert hps["output_length"] == 1, "pp_likelihood should not \
            work with output length larger than one"

        self._add_placeholder()
        self._build_graph()
        if self.verbose:
            print "Graph Output", self._output
        tv = tf.GraphKeys.TRAINABLE_VARIABLES
        self._trainable_variable = tf.get_collection(tv, scope=None)
        self._add_train()
        if not logdir:
            self.log_dir = "./log/log" + strftime("%m%d:%H%M", gmtime())
        else:
            self.log_dir = "./log/" + logdir
        self._add_logger()
        self.count_params()
        print "-" * 30

    @staticmethod
    def _check_hps(hps):
        try:
            if hps["num_labels"] == 2:
                # no redundant output for binary classification
                hps["num_labels"] = 1
            for i in ["num_labels"]:
                assert i in hps, str(i)
        except AssertionError as e:
            print e.message, "should be in hps"
        return hps

    def _embed_signal(self, signal):
        if not self._hps["embed_signal"]:
            raise ValueError("Conflict config: embedding signal")
        return pp_embed_signal(
            signal, vocab_size=self._hps["input_channels"],
            embed_size=self._hps["embed_size"])

    def _add_placeholder(self):
        raise NotImplementedError()

    def _build_graph(self):
        raise NotImplementedError()

    def _add_logger(self):
        if not self.restore:
            self.saver = tf.train.Saver()
        train_loss = tf.summary.scalar("TrainLoss", self._loss)
        train_acc = tf.summary.scalar("TrainAccuracy", self._accuracy)
        test_loss = tf.summary.scalar("TestLoss", self._loss)
        test_acc = tf.summary.scalar("TestAccuracy", self._accuracy)
        self._eval_summary = tf.summary.merge([test_loss, test_acc])
        self._merged_summary_op = tf.summary.merge([train_loss, train_acc])
        self._summary_writer = tf.summary.FileWriter(
            self.log_dir, graph=tf.get_default_graph())
        self.graph = tf.train.export_meta_graph(self.log_dir + "/model.meta")
        with open(self.log_dir + "/config.json", "w") as fout:
            json.dump(self._hps, fout, indent=4)

    def count_params(self):
        counter = 0
        for tv in self._trainable_variable:
            sub_counter = 1
            for i in tv.get_shape().as_list():
                sub_counter *= i
            counter += sub_counter
        print "Totally %d Params in the Model" % counter
        return counter

    def log_trainable_variables(self, sess, out_path_prefix):
        dir_path = re.findall("(.+/)", out_path_prefix)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for tv in self._trainable_variable:
            value = sess.run(tv)
            print "saving", tv.name, out_path_prefix
            np.save(out_path_prefix + re.sub("/|:", "_", tv.name), value)

    def restore_model(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint(self.log_dir))

    def add_customized_layers(self, last_state, layer_configs):
        layer_idx = 1
        for layer in layer_configs:
            print last_state

            if "act_fn" in layer and layer["act_fn"] is None:
                act_fn = None
            else:
                act_fn = getattr(tf.nn, layer["act_fn"], tf.nn.relu)
            if layer["type"] == "conv":
                last_state = tf.layers.conv1d(
                    last_state,
                    filters=layer["filter"],
                    kernel_size=layer["kernel_size"],
                    padding=layer.setdefault("padding", "same"),
                    activation=act_fn,
                    strides=layer.setdefault("stride", 1),
                    kernel_initializer=tf.truncated_normal_initializer(
                        mean=0.0, stddev=0.01),
                )
            elif layer["type"] == "max_pool":
                dim = len(last_state.get_shape().as_list())
                pooling = tf.layers.max_pooling1d if dim == 3 \
                    else tf.layers.max_pooling2d
                last_state = pooling(
                    last_state, layer["pool_size"], layer["stride"])

            elif layer["type"] == "fully_connected":
                if len(last_state.get_shape().as_list()) != 2:
                    _, l, c = last_state.get_shape().as_list()
                    last_state = tf.reshape(
                        last_state, shape=[self._hps["batch_size"], l * c])
                last_state = tf.contrib.layers.fully_connected(
                    last_state, layer["num_output"],
                    activation_fn=act_fn)
                if layer.setdefault("expand_dim", False):
                    last_state = tf.expand_dims(last_state, axis=-2)

            elif layer["type"] == "pp_output":
                if self.loss_func.__name__ != "pp_likelihood":
                    raise ValueError("Using pp_output layer with %s "
                                     % self.loss_func.__name__ +
                                     "loss is unreasonable")

                last_state = pp_output_layer(
                    last_state, self._hps["input_channels"], self._dt)

            if self.hist:
                tf.summary.histogram(layer["type"] + "_" + str(layer_idx),
                                     last_state)
                layer_idx += 1

        return last_state

    def _add_train(self):
        if self.restore:
            self._loss = tf.get_collection("loss")[0]
            self._train = tf.get_collection("train")[0]
            self._prediction = tf.get_collection("prediction")[0]
            self._accuracy = tf.get_collection("accuracy")[0]
            return

        self._trainable_variable = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.device('/gpu:0'):
            if self.loss_func.__name__ == "weighted_l2":
                self._prediction = tf.zeros_like(self._target)
                self._accuracy = tf.zeros([], dtype=tf.float32)
                self._loss = self.loss_func(
                    labels=self._target,
                    output=self._output,
                    mask=self._mask)

            elif self.loss_func.__name__ == "pp_likelihood":
                # actually prediction of point process is done beyond tf side
                self._prediction = self._output
                # just leave prediction as output to avoid incompatibility

                self._loss = self.loss_func(
                    markers=self._target,
                    output=self._output)
                self._accuracy = tf.constant(0, name="DummyAccuracy")

            elif self._hps["num_labels"] == 1:
                self._prediction = tf.cast(
                    tf.greater(self._output, tf.constant(0.5)),
                    tf.float32
                )
                correct_prediction = tf.equal(self._prediction, self._target)
                self._accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
                self._loss = self.loss_func(
                    labels=self._target, output=self._output, num_labels=1)
            else:
                self._prediction = tf.argmax(self._output, axis=-1,
                                             name="Predictions")
                correct_prediction = tf.equal(
                    self._prediction, tf.argmax(self._target, axis=-1))
                self._accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
                self._loss = self.loss_func(
                    labels=self._target, output=self._output)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            gvs = optimizer.compute_gradients(self._loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var
                          in gvs]
            self._train = optimizer.apply_gradients(
                capped_gvs, global_step=self.global_step)

            if self.hist:
                for gv in capped_gvs:
                    tf.summary.histogram(gv[0].name, gv[0])

            tf.add_to_collection("loss", self._loss)
            tf.add_to_collection("train", self._train)
            tf.add_to_collection("prediction", self._prediction)
            tf.add_to_collection("accuracy", self._accuracy)

        if self.verbose:
            print "\n\nTrainable Variables in the model are:"

        for x in self._trainable_variable:
            if self.hist:
                tf.summary.histogram(x.name, x)
            if self.verbose:
                print "\t", x

    @staticmethod
    def initialize(sess):
        sess.run(tf.global_variables_initializer())
        print "Variables Initialized for the Network"

    def _update_dict(self, feed_dict, **kwargs):
        if self.loss_func.__name__ == "weighted_l2":
            feed_dict.update({self._mask: kwargs["mask"]})
        elif self.loss_func.__name__ == "pp_likelihood":
            feed_dict.update({self._dt: kwargs["dt"]})


class TwoHot(object):
    def __init__(self, twohots, truncate_length, delta):
        self.twohots = twohots
        if not hasattr(truncate_length, "__iter__"):
            truncate_length = [-truncate_length, truncate_length]
        self.truncate_length = truncate_length
        if twohots:
            self.thvec_len = int((truncate_length[1] - truncate_length[0]) /
                                 delta) + 2
            self.start_value = (sum(truncate_length) / 2.0 -
                                (self.thvec_len - 1) / 2.0 * delta)
            assert delta < (truncate_length[1] -
                            truncate_length[0]), "Delta too large"
            self.delta = delta
        else:
            self.thvec_len = 1

    def scalar2id_value(self, diff):
        if not self.truncate_length[0] <= diff <= self.truncate_length[1]:
            return []
        index = int((diff - self.start_value) / self.delta)
        value = (diff - (self.start_value + self.delta * index)) / self.delta
        if value < 1e-10:
            return [(index, 1)]
        else:
            return [(index, 1 - value), (index + 1, value)]

    def scalar2vec(self, diff):
        pairs = self.scalar2id_value(diff)
        vec = [0 for _ in range(self.thvec_len)]
        for idx, val in pairs:
            vec[idx] = val
        return vec
