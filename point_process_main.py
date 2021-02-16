import tensorflow as tf
import json
import sys
import numpy as np
from Models.ccnn import CCNN as Model
from Data.NeuralHawkesData.NHBatcher import PointProcessBatcher as Batcher
from Models.loss_funcs import pp_likelihood
from Models.pp_layers import PP_mse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

config_file = sys.argv[1]

with open(config_file) as fp:
    hps = json.load(fp)

# set up session
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.allow_soft_placement = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

# set up batcher
batcher_config = hps
model_config = hps
batcher = Batcher(batcher_config)
model_config["input_length"] = batcher.input_length
model_config["input_channels"] = batcher.input_channel
model_config["num_labels"] = batcher.output_channel
model_config["output_length"] = 1
if "layers" in model_config:
    model_config["layers"][-1]["filter"] = batcher.output_channel

# set up model
offset = hps["resample_config"]["bias_kernel"] - 1
hps["resample_config"]["output_lengths"] = [hps["input_length"] - offset]


def process_itvl(itvl_in, *args, **kwargs):
    intervals = [itvl_in, []]
    for itin in itvl_in:
        uniform_out = [itin[-1] - idx for idx in
                       range(len(itin) - offset, 0, -1)]
        intervals[1].append(uniform_out)
    return intervals


logdir = "TPP_CCNN/" + model_config["logdir"]


# for extra input for label
class PPModel(Model):
    @staticmethod
    def _check_hps(hps):
        """
        remove classification num of labels check
        """
        return hps

    def train(self, sess, interval, signal, target, *args, **kwargs):
        output, lambda_0, w_t, gs, acc, loss, _, smy = self._run(
            sess, signal, interval, target,
            self._output,
            tf.get_collection("lambda_0")[0],
            tf.get_collection("w_t")[0],
            self.global_step,
            self._accuracy,
            self._loss,
            self._train,
            self._merged_summary_op,
            dt=kwargs["dt"]
        )
        self._summary_writer.add_summary(smy, global_step=gs)

        if 'label' in kwargs:
            predicted_label = np.argmax(output[0], axis=-1)
            true_label = np.argmax(kwargs['label'], axis=-1).reshape([-1])
            accuracy = sum(1 for p, t in zip(true_label, predicted_label)
                           if p == t)
            accuracy /= float(len(true_label))
            return gs, accuracy, PP_mse(lambda_0, w_t, kwargs['dt']), loss
        return gs, PP_mse(lambda_0, w_t, kwargs["dt"]), loss

    def evaluate(self, sess, interval, signal, target, *args, **kwargs):
        output, lambda_0, w_t, gs, acc, loss, smy = self._run(
            sess, signal, interval, target,
            self._output,
            tf.get_collection("lambda_0")[0],
            tf.get_collection("w_t")[0],
            self.global_step,
            self._accuracy,
            self._loss,
            self._eval_summary,
            dt=kwargs["dt"]
        )
        self._summary_writer.add_summary(smy, global_step=gs)

        if 'label' in kwargs:
            predicted_label = np.argmax(output[0], axis=-1)
            true_label = np.argmax(kwargs['label'], axis=-1).reshape([-1])
            accuracy = sum(1 for p, t in zip(true_label, predicted_label)
                           if p == t)
            accuracy /= float(len(true_label))
            return gs, accuracy, PP_mse(lambda_0, w_t, kwargs['dt'])
        return gs, PP_mse(lambda_0, w_t, kwargs["dt"]), loss

    def predictions(self, sess, interval, signal, target, *args, **kwargs):
        output, lambda_0, w_t, gs, acc, loss, smy = self._run(
            sess, signal, interval, target,
            self._output,
            tf.get_collection("lambda_0")[0],
            tf.get_collection("w_t")[0],
            self.global_step,
            self._accuracy,
            self._loss,
            self._eval_summary,
            dt=kwargs["dt"]
        )

        mse, dt_prediction = PP_mse(lambda_0, w_t, kwargs['dt'],
                                    return_predictions=True)
        intensity = output[1]
        return dt_prediction, intensity


model = PPModel(model_config, verbose=True,
                loss_func=pp_likelihood, logdir=logdir)
if model_config["restore"]:
    model.restore_model(sess)
else:
    model.initialize(sess)


def change_data(time_in, signals, time_out, labels):
    """
    :param time_in: timestamp, [batch, input_length]
    :param signals: one-hot labels, [batch, input_length, num_labels]
    :param time_out: timestamp, [batch, output_length]
    :param labels: one-hot labels, [batch, output_length, num_labels]
    :return:
        itvl: [batch, layers, length], for CCNN,
              [batch, input_length] for others.
              output dt not included
        signals: as signals
        dts: [batch, 1, output_time - last_input_time]
        markers: as labels
    """

    itvl = process_itvl(time_in)
    # if model is CCNN, it will change itvl from [batch, input_length]
    # to [batch, num_ccnn_layers, input_length/layer_output_len]
    # print itvl

    dt = list()
    for tin, tout in zip(time_in, time_out):
        dt.append([tout[0] - tin[-1]])
    return itvl, signals, dt, labels


def main():
    best = (100, 100, 100)
    for epoch in range(100):
        train_acc = 0
        train_mse = 0
        train_size = 0
        for time_in, signals, time_out, labels in batcher.iter_train(
                model_config["batch_size"]):

            itvl, signals, dt, markers = change_data(time_in, signals, time_out,
                                                     labels)
            try:
                gs, accuracy, loss = model.train(
                    sess, interval=itvl, signal=signals, target=markers, dt=dt,
                    label=markers)
                print(gs)
                # todo: delete
                exit()
                if gs % 10 == 0:
                    print "Global Step %d | Train Accuracy %f | Loss = %f" % \
                          (gs, accuracy, loss)
                train_acc += accuracy
                train_mse += loss
                train_size += 1
            except Exception:
                continue
        train_acc /= train_size
        train_mse /= train_size
        train_mse = np.sqrt(train_mse) * batcher.norm_factor

        # model.log_trainable_variables(sess, "log/" + logdir + "/")

        test_acc = 0
        test_mse = 0
        test_size = 0
        for time_in, signals, time_out, labels in batcher.iter_test(
                model_config["batch_size"]):
            itvl, signals, dt, markers = change_data(time_in, signals, time_out,
                                                     labels)
            try:
                gs, accuracy, loss = model.evaluate(
                    sess, interval=itvl, signal=signals, target=markers, dt=dt,
                    label=markers)
                print "Global Step %d | Test Accuracy %f | Loss = %f" % \
                      (gs, accuracy, loss)
                test_acc += accuracy
                test_mse += loss
                test_size += 1
            except Exception:
                continue
        test_acc /= test_size
        test_mse /= test_size
        test_mse = np.sqrt(test_mse) * batcher.norm_factor

        dev_acc = 0
        dev_mse = 0
        dev_size = 0
        for time_in, signals, time_out, labels in batcher.iter_dev(
                model_config["batch_size"]):
            itvl, signals, dt, markers = change_data(time_in, signals, time_out,
                                                     labels)
            try:
                gs, accuracy, loss = model.evaluate(
                    sess, interval=itvl, signal=signals, target=markers, dt=dt,
                    label=markers)
                print "Global Step %d | Dev Accuracy %f | Loss = %f" % \
                      (gs, accuracy, loss)
                dev_acc += accuracy
                dev_mse += loss
                dev_size += 1
            except ValueError:
                continue
        dev_acc /= dev_size
        dev_mse /= dev_size
        dev_mse = np.sqrt(dev_mse) * batcher.norm_factor
        print "[SUM] Epoch %d" % epoch
        print "[SUM][Accuracy] | Train %f | Test %f | Dev %f" % (
            train_acc * 100, test_acc * 100, dev_acc * 100)
        print "[SUM][RMSE] | Train %f | Test %f | Dev %f" % (
            train_mse, test_mse, dev_mse)

        if test_mse < best[1]:
            best = (train_mse, test_mse, dev_mse)

            intensity_record = []
            dt_record = []
            for time_in, signals, time_out, labels in batcher.iter_test(
                    model_config["batch_size"]):
                itvl, signals, dt, markers = change_data(time_in, signals,
                                                         time_out, labels)
                try:
                    dt_pred, intensity = model.predictions(
                        sess, interval=itvl, signal=signals, target=markers,
                        dt=dt, intensity=True)
                    intensity_record.extend(intensity)
                    dt_record.extend(dt_pred)
                except ValueError:
                    continue
            np.save("log/" + logdir + "/intensity.npy",
                    np.array(intensity_record))
            np.save("log/" + logdir + "/dt_prediction.npy", np.array(dt_record))

    print "[SUM] Best Model| Train %f | Test %f | Dev %f" % best


if __name__ == "__main__":
    main()
