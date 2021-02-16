import tensorflow as tf
import json
import sys
from Models.loss_funcs import l2_norm_loss
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_name = sys.argv[1]
exp_name = sys.argv[2]
data_name = sys.argv[2]
config_file = sys.argv[3]


with open(config_file) as fp:
    hps = json.load(fp)


# set up session
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.allow_soft_placement = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)


# set up batcher
batcher_config = None
if exp_name == "sine":
    from Data.sinusoid_signal import PredSignal as Batcher
    batcher_config = hps["signal_config"]
    batcher_config["output_length"] = hps["output_length"]
    batcher_config["input_length"] = hps["input_length"]
elif exp_name in ["glass", "lorenz"]:
    from Data.prediction.ChaoticBatcher import MGlass as Batcher
else:
    raise AttributeError("no such data set {}" % exp_name)
    exit()

if batcher_config is None:
    batcher_config = hps

model_config = hps
batcher = Batcher(batcher_config)
model_config["input_length"] = batcher.input_length
model_config["input_channels"] = batcher.input_channel
model_config["output_length"] = batcher.output_length
model_config["num_labels"] = batcher.output_channel


# set up model
if model_name == "CCNN":
    offset = hps["resample_config"]["bias_kernel"] - 1
    hps["resample_config"]["output_lengths"] = [hps["input_length"] - offset]
    from Models.ccnn import CCNN as Model

    def process_itvl(itvl_in, itvl_out, diff=True, *args, **kwargs):
        intervals = [itvl_in, []]
        for itin, itout in zip(itvl_in, itvl_out):
            uniform_out = [itout[0] - 1 - idx for idx in
                           range(len(itin) - offset, 0, -1)]
            intervals[1].append(uniform_out)
            ldiff = itin[-1] - itout[0]
        if diff:
            return intervals, ldiff
        return intervals

elif model_name == "CNN":
    from Models.cnn import CNN as Model

    def process_itvl(itvl_in, itvl_out, diff=True):
        if diff:
            return itvl_in, itvl_out[0][0] - itvl_in[0][-1]
        return itvl_in

elif model_name in ["CNT", "ICNN", "RNN"]:

    def process_itvl(itvl_in, itvl_out, diff=True, *args, **kwargs):
        intervals = list()
        for itin, itout in zip(itvl_in, itvl_out):
            try:
                itin = itin.tolist()
            except AttributeError:
                pass
            itin.append(itout[-1])
            intervals.append(itin)
        if diff:
            return intervals, intervals[0][-1] - intervals[0][-2]
        return intervals

    if model_name == "CNT":
        from Models.cnt import CNT as Model
    elif model_name == "RNN":
        from Models.rnn import SimpleRNN as Model
    else:
        from Models.cnn import ICNN as Model

else:
    exit()


logdir = exp_name.upper() + "/" + model_name.upper() + "/" + model_config["logdir"]

model = Model(model_config, verbose=True, loss_func=l2_norm_loss, logdir=logdir)
if model_config["restore"]:
    model.restore_model(sess)
else:
    model.initialize(sess)


def main():
    gs = 0
    while gs < 5000:
        itvl_in, signals, itvl_out, labels = batcher.next_train(
            model_config["batch_size"])
        # itvl are timestamps
        # print itvl_in

        itvl = process_itvl(itvl_in, itvl_out, diff=False)
        gs, accuracy, loss = model.train(
            sess, interval=itvl, signal=signals, target=labels)
        if gs % 10 == 0:
            print "Global Step %d | Train Accuracy %f | Loss = %f" % \
                  (gs, accuracy, loss)

        if gs % 1000 == 0:
            model.log_trainable_variables(sess, "log/" + logdir + "/")
            # exit()
            diffs = list()
            losses = list()
            for _ in range(1000):
                itvl_in, signals, itvl_out, labels = batcher.next_test(
                    model_config["batch_size"])
                itvl, diff = process_itvl(itvl_in, itvl_out, diff=True)
                diffs.append(diff)

                gs, accuracy, loss = model.evaluate(
                    sess, interval=itvl, signal=signals, target=labels)
                print "Global Step %d | Test Accuracy %f | Loss = %f" % \
                      (gs, accuracy, loss)
                losses.append(loss)


if __name__ == "__main__":
    main()
