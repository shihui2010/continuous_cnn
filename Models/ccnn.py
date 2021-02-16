from model import Model, TwoHot
import math
import sys
import tensorflow as tf
from loss_funcs import softmax_cross_entropy_with_logits_loss


class TimeTransform(TwoHot):
    def __init__(self, in_itvl, out_itvl, signal, r_kernel, b_kernel,
                 truncate_length, twohots, delta, concat):
        super(TimeTransform, self).__init__(twohots, truncate_length, delta)

        self.in_itvl = in_itvl
        self.out_itvl = out_itvl
        self.signal = signal
        self.batch_size = len(in_itvl)
        self.seq_out = len(out_itvl[0])
        self.seq_in = len(in_itvl[0])
        self.concat = concat

        self.r_kernel = r_kernel
        self.b_kernel = b_kernel

        self.begin_rdim_idx = list()
        # for 3D matrix [batch, sout, sin]
        self.last_rdim_idx = list()
        # for fully connected layer, [Nr, idx in conv * th_vec]
        self.r_values = list()
        self.Nr = 0

        self.Nb = 0
        self.bdim_idx = list()
        self.b_values = list()

        if concat:
            self.vec_len = self.thvec_len + len(self.signal[0][0])
        else:
            self.vec_len = self.thvec_len

        self.transform()

        assert self.Nr == len(self.begin_rdim_idx), "Incompatible"

    def get_diff(self, bid, sin_id, sout_id):
        return self.out_itvl[bid][sout_id] - self.in_itvl[bid][sin_id]

    def truncate(self, bid, sin_id, sout_id):
        return self.get_diff(bid, sin_id, sout_id) >= self.truncate_length

    # @profile
    def transform(self):
        """
        modifies idx list and values for construction of SparseTensor
        begin_dim_idx:
            3D list [[batch_idx, seq_in_idx, seq_out_idx], ...]
        value:
            1D list [value1, ...]
        shape:
            [batch_size, seq_out_len, seq_in_len, last_dim]
        """

        assert len(self.in_itvl) == len(self.out_itvl), "Batch size not equal"

        for bid in range(self.batch_size):
            last_start_idx = 0
            for sout_id in range(self.seq_out):
                this_start_idx = None
                min_idx, min_abs_value = -1, sys.maxint
                for sin_id in range(last_start_idx, self.seq_in):
                    diff = self.get_diff(bid, sin_id, sout_id)
                    if diff >= self.truncate_length[1]:
                        continue
                    elif diff < self.truncate_length[0]:
                        break

                    # else: for valid time diff

                    # update start idx
                    if this_start_idx is None:
                        this_start_idx = sin_id
                    # update closest ti and tj
                    if abs(diff) < min_abs_value:
                        min_idx = sin_id
                        min_abs_value = abs(diff)

                    if self.twohots:
                        id_value = self.scalar2id_value(diff)
                        offset = self.thvec_len
                    else:
                        id_value = [(0, diff)]
                        offset = 1

                    if self.concat:  # concat signal for rnet
                        signal_value = self.signal[bid][sin_id]
                        for concat_idx, s_channel in enumerate(signal_value):
                            id_value.append((offset + concat_idx, s_channel))

                    start_idx = max(0, sin_id - self.r_kernel / 2)
                    end_idx = min(self.seq_in, sin_id + (self.r_kernel + 1) / 2)
                    for kid in range(start_idx, end_idx):
                        diff = self.get_diff(bid, kid, sout_id)
                        if diff >= self.truncate_length[1]:
                            continue
                        elif diff < self.truncate_length[0]:
                            break
                        offset = kid - sin_id
                        last_dim_id = ((self.r_kernel / 2 + offset)
                                       * self.vec_len)
                        for idx, value in id_value:
                            self.last_rdim_idx.append(
                                [self.Nr + offset, idx + last_dim_id])
                            self.r_values.append(value)

                    self.begin_rdim_idx.append([bid, sout_id, sin_id])
                    self.Nr += 1

                b_start = max(0, min_idx - self.b_kernel / 2)
                b_end = min(self.seq_in, min_idx + (self.b_kernel + 1)/ 2)
                for sin_id in range(b_start, b_end):
                    diff = self.get_diff(bid, sin_id, sout_id)
                    if diff >= self.truncate_length[1]:
                        continue
                    elif diff < self.truncate_length[0]:
                        break

                    if self.twohots:
                        id_value = self.scalar2id_value(diff)
                        offset = self.thvec_len
                    else:
                        id_value = [(0, diff)]
                        offset = 1

                    if self.concat: # concat signal value for bnet
                        signal_value = self.signal[bid][sin_id]
                        for concat_idx, s_channel in enumerate(
                                signal_value):
                            id_value.append((offset + concat_idx, s_channel))

                    offset = sin_id - min_idx
                    last_dim_id = (self.b_kernel / 2 + offset) * self.vec_len
                    for idx, value in id_value:
                        self.bdim_idx.append(
                            [self.Nb, idx + last_dim_id])
                        self.b_values.append(value)

                self.Nb += 1

                if this_start_idx is not None:
                    last_start_idx = this_start_idx


class CCNN(Model):
    def __init__(self, hps, input_channels=1,
                 name='weight_net', verbose=True, logdir=None,
                 loss_func=softmax_cross_entropy_with_logits_loss):
        self.restore = hps['restore']

        if self.restore:
            last_checkpoint = tf.train.latest_checkpoint("./log/" + logdir)
            meta_filename = last_checkpoint + ".meta"
            self.saver = tf.train.import_meta_graph(meta_filename)
        else:
            self.learning_rate = hps['learning_rate']

        self.values = list()
        self.idxs = list()
        self.origin_idxs = list()
        self.verbose = verbose

        self.model_name = name
        self.input_channels = input_channels
        self._rhps = hps["resample_config"]

        if "kernel_on_t" not in self._rhps:
            self._rhps["kernel_on_t"] = True  # default setting

        self.set_activation(hps)
        super(CCNN, self).__init__(
            hps, logdir=logdir,  loss_func=loss_func)

    def set_activation(self, hps):
        self.act_main = getattr(
            tf.nn, hps["activation_fn"], tf.nn.relu)
        self.act_weight = getattr(
            tf.nn, self._rhps["activation_fn"], tf.nn.relu)

    def restore_placeholder(self):
        self._signal_input = tf.get_collection("Signals")[0]
        self._target = tf.get_collection("Targets")[0]
        self._r_begin_idxs = tf.get_collection("RBeginIDX")

        self._tdflat_indices = tf.get_collection("TDiffFlatIndex")
        self._tdflat_values = tf.get_collection("TDiffFlatValue")
        self._tdflat_lens = tf.get_collection("TDiffFlatLen")

        if self._rhps['bias_on_t'] and self._hps['main_net_bias']:
            self._tddiag_indices = tf.get_collection("TDiffDiagIndex")
            self._tddiag_values = tf.get_collection("TDiffDiagValue")
            self._tddiag_lens = tf.get_collection("TDiffDiagLen")

    def set_thvec_length(self):
        th_length = 1
        if self._hps['twohots']:
            th_length += int((self._hps["truncate_length"][1] -
                              self._hps["truncate_length"][0])
                            / self._hps["delta"]) + 1

        if self._hps["concat"]:
            # to concat signal
            th_length += self._hps["input_channels"]

        self.thvec_r = self._rhps["rnet_kernel"] * th_length
        self.thvec_b = self._rhps["bias_kernel"] * th_length
        print "thvec r: %d, thvec b: %d" % (self.thvec_r, self.thvec_b)

    def _add_placeholder(self):
        if self.restore:
            self.restore_placeholder()
            return

        with tf.device('/cpu:0'):
            self.set_thvec_length()
            self._time_diff_flats = list()
            self._r_begin_idxs = list()
            self._tdiff_diag_flats = list()
            seq_lens = [self._hps['input_length']]
            seq_lens.extend(
                self._rhps['output_lengths'])

            self._tdflat_indices = list()
            self._tdflat_values = list()
            self._tddiag_indices = list()
            self._tddiag_values = list()
            self._tdflat_lens = list()
            self._tddiag_lens = list()

            for _ in self._rhps["output_lengths"]:
                tdflat_index = tf.placeholder(
                    shape=[None, 2],
                    dtype=tf.int64, name="td_flat_index")
                tdflat_value = tf.placeholder(
                    shape=[None],
                    dtype=tf.float32, name="td_flat_value")
                tdflat_len = tf.placeholder(
                    shape=[2], dtype=tf.int64, name="td_flat_len")

                time_diff_flat = tf.SparseTensor(
                    values=tdflat_value,
                    indices=tdflat_index,
                    dense_shape=tdflat_len)

                tf.add_to_collection("TDiffFlatValue", tdflat_value)
                tf.add_to_collection("TDiffFlatIndex", tdflat_index)
                tf.add_to_collection("TDiffFlatLen", tdflat_len)
                self._tdflat_indices.append(tdflat_index)
                self._tdflat_values.append(tdflat_value)
                self._tdflat_lens.append(tdflat_len)

                self._time_diff_flats.append(time_diff_flat)

                r_begin_idx = tf.placeholder(
                    dtype=tf.int64, shape=[None, 3], name="r_begin_idx")
                tf.add_to_collection("RBeginIDX", r_begin_idx)
                self._r_begin_idxs.append(r_begin_idx)

                tddiag_flat_index = tf.placeholder(
                    shape=[None, 2],
                    dtype=tf.int64, name="td_diag_index")
                tddiag_flat_value = tf.placeholder(
                    shape=[None], dtype=tf.float32,
                    name="td_diag_value")
                tddiag_len = tf.placeholder(
                    shape=[2], dtype=tf.int64, name="td_diag_len")

                tdiff_diag_flat = tf.SparseTensor(
                    values=tddiag_flat_value, indices=tddiag_flat_index,
                    dense_shape=tddiag_len)

                tf.add_to_collection("TDiffDiagValue", tddiag_flat_value)
                tf.add_to_collection("TDiffDiagIndex", tddiag_flat_index)
                tf.add_to_collection("TDiffDiagLen", tddiag_len)
                self._tddiag_indices.append(tddiag_flat_index)
                self._tddiag_values.append(tddiag_flat_value)
                self._tddiag_lens.append(tddiag_len)

                self._tdiff_diag_flats.append(tdiff_diag_flat)

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
        self.add_adaptive_initializer()

    def add_adaptive_initializer(self):
        b_stddev = self._rhps["bnet_winit"] / math.sqrt(self.thvec_b)
        self.b_normal = tf.truncated_normal_initializer(stddev=b_stddev)
        b_span = self._rhps["bnet_binit"] / math.sqrt(self.thvec_b)
        if self._rhps["bnet_bias"]:
            self.b_uniform = tf.random_uniform_initializer(
                minval=-b_span, maxval=b_span)
        else:
            self.b_uniform = None

        r_stddev = self._rhps["rnet_winit"] / math.sqrt(self.thvec_r)
        self.r_normal = tf.truncated_normal_initializer(stddev=r_stddev)
        r_span = self._rhps["rnet_binit"] / math.sqrt(self.thvec_r)
        self.r_uniform = tf.random_uniform_initializer(
            minval=-r_span, maxval=r_span)

    def sparse_fc(self, inputx, shape, num_output, weights_initializer,
                  bias_initializer, activation_fn):
        """
        :param inputx: N-D tensor, N > 1
        :param shape: shape of inputx
        inputx # of channels = kernel length * thvec_len (+ signal_channels)
        """
        weight = tf.Variable(
            weights_initializer(shape=[shape[-1], num_output]),
            name="weights")
        if isinstance(inputx, tf.SparseTensor):
            reshape = tf.sparse_reshape(inputx, shape=[-1, shape[-1]])
            output = tf.sparse_tensor_dense_matmul(reshape, weight)
        else:
            reshape = tf.reshape(inputx, shape=[-1, shape[-1]])
            output = tf.matmul(reshape, weight)
        if bias_initializer is not None:
            bias = tf.Variable(
                bias_initializer(shape=[num_output]),
                name="bias")
            output += bias
        shape[-1] = num_output
        if activation_fn is None:
            return tf.reshape(output, shape)
        return activation_fn(tf.reshape(output, shape=shape))

    def embed_sparse_fc(self, inputx, shape, num_output, weights_initializer,
                        bias_initializer, act_fn, num_splits):
        """
        Assume inputx might be [twohot_t1, signal_t1, twohot_t2, signal_t2...]
        embed each signal before feeding the long vector to fully connected
        :param inputx: [th, s, th, s,...]
        :param shape: shape of inputx
        :param num_splits: should be either rnet_kernel or bias_kernel
        :return: embeded and fully_connected inputx
        """
        input_dim = shape[-1]
        assert input_dim % num_splits == 0, "can not evenly split"
        dim = input_dim / num_splits
        dim_s = self._hps["input_channels"]
        inputx = tf.sparse_to_dense(inputx.indices,
                                    inputx.dense_shape,
                                    inputx.values,
                                    validate_indices=False)
        shards = tf.split(inputx, num_splits, axis=-1)
        merge = []
        for shard in shards:
            t, s = tf.split(shard, [dim - dim_s, dim_s], axis=-1)
            merge.append(t)
            merge.append(self._embed_signal(s))
        embeded_input = tf.concat(merge, axis=-1)
        shape[-1] -= num_splits * (dim_s - self._hps["embed_size"])

        return self.sparse_fc(embeded_input, shape, num_output,
                              weights_initializer, bias_initializer, act_fn)

    def sparse_stack(self, inputx, shape, num_output, hidden_units, layer_nums,
                     weights_initializer, bias_initializer, activation_fn,
                     num_splits):
        if layer_nums == 1:
            if self._hps["concat"] and self._hps["embed_signal"]:
                return self.embed_sparse_fc(
                    inputx, shape, num_output, weights_initializer,
                    bias_initializer, None, num_splits)
            return self.sparse_fc(
                inputx, shape, num_output,
                weights_initializer, bias_initializer, None)

        if self._hps["concat"] and self._hps["embed_signal"]:
            output = self.embed_sparse_fc(
                inputx, shape, hidden_units, weights_initializer,
                bias_initializer, activation_fn, num_splits)
        else:
            output = self.sparse_fc(
                inputx, shape, hidden_units,
                weights_initializer, bias_initializer, activation_fn)

        for layer_index in range(1, layer_nums - 1):
            output = tf.layers.dense(
                output, hidden_units,
                kernel_initializer=weights_initializer,
                bias_initializer=bias_initializer,
                activation=activation_fn)

        output = tf.layers.dense(
            output, num_output,
            kernel_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            activation=None)
        return output

    @staticmethod
    def idx_to_2d(idx_3d, seq_in, seq_out, chin, chout):
        """
        :param idx_3d: 2D tensor with shape[[bid, sout_id, sin_id], ...]
        :param seq_in: int, length of in_sequence
        :param seq_out: int, length of out_sequence
        :param chin: input channels
        :param chout: output channels
        :return: 2D tensor: [[bid * sout * chout +  sout_id * chout + chout_id,
                             [bid * sin * chin + sin_id * chin + chin_id]
        """
        idx_3d = tf.reshape(tf.tile(idx_3d, [1, chin * chout]),
                            [-1, chin * chout, 3])
        bid, sout_id, sin_id = tf.unstack(idx_3d, 3, -1)
        in_channels = tf.tile(tf.range(chin, dtype=tf.int64), [chout])
        out_channels = tf.tile(
            tf.expand_dims(tf.range(chout, dtype=tf.int64), -1),
            [1, chin])
        out_channels = tf.reshape(out_channels, [-1])
        out_idx = bid * (seq_out * chout) + sout_id * chout + out_channels
        in_idx = bid * (seq_in * chin) + sin_id * chin + in_channels
        out_idx = tf.reshape(out_idx, [-1])
        in_idx = tf.reshape(in_idx, [-1])
        return tf.stack([out_idx, in_idx], 1)

    def get_rx(self, tdiff_flat, tdf_shape, begin_idx, signal,
               output_len, out_channels):
        if not self._rhps["kernel_on_t"]:
            assert not self._hps["concat"], "not supported yet"
            if self._hps["embed_signal"]:
                signal = self._embed_signal(signal)
            weights = tf.Variable(
                self.r_normal(shape=[
                    self._rhps["bias_kernel"],  # to be shape consistent
                    signal.get_shape().as_list()[-1],
                    self._rhps["output_num"]]),
                name="weights")

            conv = tf.nn.conv1d(signal, weights, 1, "VALID", name="conv_sig")
            return conv

        output_values = self.sparse_stack(
            tdiff_flat, tdf_shape, out_channels,
            self._rhps["rnet_hidden_units"], self._rhps["rnet_layer_nums"],
            self.r_normal, self.r_uniform, self.act_weight,
            self._rhps["rnet_kernel"])
        output_values = tf.reshape(output_values, [-1])

        if self._hps["embed_signal"]:
            signal = self._embed_signal(signal)

        input_len, chin = signal.get_shape().as_list()[1:]
        chout = out_channels / chin
        if not chout:
            raise ValueError("out channels can be zero")

        dense_shape = [self._hps["batch_size"] * output_len * chout,
                       self._hps["batch_size"] * input_len * chin]
        self.origin_idxs.append(begin_idx)
        begin_idx = self.idx_to_2d(begin_idx, input_len, output_len, chin, chout)
        self.values.append(output_values)
        self.idxs.append(begin_idx)

        r_matrix = tf.SparseTensor(begin_idx, output_values, dense_shape)
        # print "r_matrix dense shape", dense_shape
        tf.add_to_collection("Rnets", r_matrix)
        self.r_matrix = r_matrix


        signal = tf.reshape(signal,
                            [self._hps["batch_size"] * input_len * chin, 1])
        # print "signal", signal.get_shape().as_list()
        # print "rmatrix", r_matrix.dense_shape
        product2d = tf.sparse_tensor_dense_matmul(r_matrix, signal)
        batch_rx = tf.reshape(product2d,
                              [self._hps["batch_size"], output_len, chout])
        return batch_rx

    def add_layer(self, tdiff_flat, begin_idx, signal, tdiff_diag_flat,
                  output_len, out_channels, scope):
        """
        compute R(ti - tj) * X
        :param time_diff: [batch_size, seq_out, seq_in + kernel - 1, last_dim]
        :param batch_last_state: [batch_size, seq_in, in_channels]
        :param mask: [batch_size, seq_out, seq_in]
        """
        with tf.variable_scope(scope + "_rnet"):
            batch_rx = self.get_rx(
                tdiff_flat, [-1, self.thvec_r],
                begin_idx, signal, output_len, out_channels)
            self.batch_rx = batch_rx
            if self.hist:
                tf.summary.histogram("batch_rx", self.batch_rx)

        if self._hps['main_net_bias']:
            chout = batch_rx.get_shape().as_list()[-1]
            if self._rhps['bias_on_t']:
                with tf.variable_scope(scope + "_bnet"):
                    bias = self.sparse_stack(
                        tdiff_diag_flat,
                        [-1, self.thvec_b], chout,
                        hidden_units=self._rhps["bnet_hidden_units"],
                        layer_nums=self._rhps["bnet_layer_nums"],
                        weights_initializer=self.b_normal,
                        bias_initializer=self.b_uniform,
                        activation_fn=self.act_weight,
                        num_splits=self._rhps["bias_kernel"])
                bias = tf.reshape(
                    bias, [self._hps["batch_size"], output_len, chout])
                tf.add_to_collection("Bnet", bias)
                batch_rx = tf.add(bias, batch_rx, name="BiasAddedBatchRX")
            else:
                bias = tf.Variable(
                    tf.truncated_normal(shape=[chout], stddev=0.01, mean=0.1),
                    trainable=True)
                batch_rx = tf.nn.bias_add(
                    batch_rx, bias, name="BiasAddedBatchRX")
        self.batch_rx = batch_rx
        if self.hist:
            tf.summary.histogram("BiasAddedBatchRX", batch_rx)
        return batch_rx

    def _build_graph(self):
        if self.restore:
            self._output = tf.get_collection("Output")[0]
            return

        with tf.device("/gpu:0"):
            last_state = self._signal_input
            if self._hps["embed_signal"]:
                chin = self._hps["embed_size"]
            else:
                chin = self._hps["input_channels"]
            out_channels = chin * self._rhps["output_num"]
            for idx, r_begin in enumerate(self._r_begin_idxs):
                last_state = self.add_layer(
                    self._time_diff_flats[idx], r_begin,
                    last_state, self._tdiff_diag_flats[idx],
                    self._rhps["output_lengths"][idx], out_channels,
                    "ResampleLayer_" + str(idx))
                if (idx != len(self._r_begin_idxs) - 1 or
                            len(self._hps["layers"]) != 0):
                    last_state = self.act_main(last_state)
                if self.verbose:
                    print last_state

            self._output = self.add_customized_layers(
                last_state, self._hps["layers"])
        tf.add_to_collection("Output", self._output)

    def _run(self, sess, signal, interval, target, *optional, **kwargs):
        td_flat_values = list()
        td_flat_indice = list()
        td_flat_lens = list()

        r_begin_idxs = list()

        tddiag_values = list()
        tddiag_indice = list()
        tddiag_lens = list()
        for idx in range(len(interval) - 1):
            tt = TimeTransform(
                interval[idx], interval[idx + 1], signal,
                self._rhps["rnet_kernel"], self._rhps["bias_kernel"],
                self._hps["truncate_length"],
                self._hps["twohots"], self._hps["delta"], self._hps["concat"])
            if len(tt.begin_rdim_idx) == 0:
                continue

            td_flat_values.append(tt.r_values)
            td_flat_indice.append(tt.last_rdim_idx)
            td_flat_lens.append([tt.Nr, tt.vec_len * tt.r_kernel])
            r_begin_idxs.append(tt.begin_rdim_idx)

            if self._rhps['bias_on_t']:
                tddiag_indice.append(tt.bdim_idx)
                tddiag_values.append(tt.b_values)
                tddiag_lens.append([tt.Nb, tt.vec_len * tt.b_kernel])

        feed_dict = {i: d for (i, d)
                     in zip(self._tdflat_values, td_flat_values)}
        feed_dict.update({i: d for (i, d)
                          in zip(self._tdflat_indices, td_flat_indice)})
        feed_dict.update({i: d for (i, d) in
                          zip(self._tdflat_lens, td_flat_lens)})

        feed_dict.update({i: d for (i, d)
                          in zip(self._r_begin_idxs, r_begin_idxs)})
        if self._rhps['bias_on_t']:
            feed_dict.update({i: d for (i, d) in
                              zip(self._tddiag_values, tddiag_values)})
            feed_dict.update({i: d for (i, d) in
                              zip(self._tddiag_indices, tddiag_indice)})
            feed_dict.update({i: d for (i, d) in
                              zip(self._tddiag_lens, tddiag_lens)})
        feed_dict[self._signal_input] = signal
        feed_dict[self._target] = target
        self._update_dict(feed_dict, **kwargs)
        try:
            result = sess.run(optional, feed_dict=feed_dict)
        except Exception:
            return [None] * len(optional)
        return result

    def train(self, sess, interval, signal, target, *optional):
        _, gs, acc, loss, smy, opt = self._run(
            sess, signal, interval, target,
            self._train,
            self.global_step,
            self._accuracy,
            self._loss,
            self._merged_summary_op,
            optional)
        self._summary_writer.add_summary(smy, global_step=gs)
        self.train_epoches += 1
        if self.train_epoches % 50 == 0:
            self.saver.save(sess,
                            self.log_dir + "/" + self.model_name + ".ckpt",
                            global_step=self.global_step)
        if len(optional) < 1:
            return gs, acc, loss
        return gs, acc, loss, opt

    def evaluate(self, sess, interval, signal, target, *optional):
        gs, acc, loss, opt = self._run(
            sess, signal, interval, target,
            self.global_step,
            self._accuracy,
            self._loss,
            optional)
        if len(optional) < 1:
            return gs, acc, loss
        return gs, acc, loss, opt
