import tensorflow as tf
from scipy import integrate
import numpy as np


def pp_output_layer(hidden_state, num_markers, dt):
    """
    :param hidden_state: from hidden layer /maybe CNN or RNN
    :param num_markers: number of types of events
    :param dt: t - t_j
    :return: P(y_{j+1}|h_j) as a vector [num_markers], f(dt) as scalar
    """
    tf.add_to_collection("input_to_pp", hidden_state)
    embed_dim = hidden_state.get_shape().as_list()[-1]
    with tf.variable_scope("pp_output"):
        V_y = tf.get_variable(
            "V_y", shape=[embed_dim, num_markers], dtype=tf.float32)
        b_y = tf.get_variable(
            "b_y", shape=[num_markers], dtype=tf.float32)

        v_t = tf.get_variable(
            "v_t", shape=[embed_dim, 1], dtype=tf.float32)
        b_t = tf.get_variable(
            "b_t", shape=[1], dtype=tf.float32)
        w_t = tf.get_variable("w_t", shape=[1], dtype=tf.float32)

        flat = tf.reshape(hidden_state, shape=[-1, embed_dim])
        tf.add_to_collection("V_y", V_y)
        tf.add_to_collection("b_y", b_y)
        tf.add_to_collection("v_t", v_t)
        tf.add_to_collection("b_t", b_t)
        multinomial = tf.nn.softmax(
            tf.nn.xw_plus_b(flat, V_y, b_y))
        plain_d0 = tf.matmul(flat, v_t) + b_t
        plain_dt = plain_d0 + w_t * dt
        lambda_d0 = tf.exp(plain_d0)
        tf.add_to_collection("lambda_0", lambda_d0)
        tf.add_to_collection("w_t", w_t)

        lambda_dt = tf.exp(plain_dt)

        likelihood = lambda_dt * tf.exp(lambda_d0 / w_t - lambda_dt / w_t)

        return multinomial, likelihood


def pp_embed_signal(signal, vocab_size, embed_size):
    """
    :param signal: should be a 3-D tensor with [batch_size, seq_len, vocab_size]
    :param vocab_size: number of types of events/markers
    :param embed_size:
    :return: embeded signal with shape [batch_size, seq_len, embed_size]
    """
    assert len(signal.get_shape().as_list()) in [2, 3], \
        ("Signal for embedding should be rank 2 or 3 but is rank" %
         len(signal.get_shape().as_list()))

    seq_len, real_vocab_size = signal.get_shape().as_list()[-2:]

    if real_vocab_size == 1 or vocab_size == 1:
        raise ValueError("Should not embedding real-value signal")
    if real_vocab_size != vocab_size:
        raise ValueError("Vocab size doesn't agree: %d v.s. %d" %
                         (real_vocab_size, vocab_size))

    with tf.variable_scope("embed_signal", reuse=tf.AUTO_REUSE):
        embed_mat = tf.get_variable(
            "embed_mat", shape=[vocab_size, embed_size])

        flatted = tf.reshape(signal, [-1, vocab_size])
        multiplied = tf.matmul(flatted, embed_mat)
        if len(signal.get_shape()) == 3:
            res = tf.reshape(multiplied, [-1, seq_len, embed_size])
        else:
            res = multiplied
    return res


def PP_mse(lambda_0, w_t, dt_data, return_predictions=False):
    """
    :param lambda_0: exp(vh + b), [batch_size], not tf.Tensor
    :param w_t: scalar, not tf.Tensor
    :param dt_data: duration to next event in dataset, [batch_size, 1]
    :return: mse for this batch
    """
    def integral(tao, lambda_0):
        if w_t != 0:
            res = tao * lambda_0 * np.exp(w_t * tao) * np.exp(lambda_0 / w_t * (
            1 - np.exp(w_t * tao)))
            if np.isnan(res):
                return 0
            return res
        else:
            res = tao * lambda_0 * np.exp(-tao * lambda_0)
            if np.isnan(res):
                return 0
            return res
    predictions = list()
    mse = 0
    for l, d in zip(lambda_0, dt_data):
        expected_dt = integrate.quad(integral, 0, np.inf, args=(l,))[0]
        predictions.append(expected_dt)
        mse += (d[0] - expected_dt) ** 2
    mse /= len(lambda_0)
    if return_predictions:
        return mse, predictions
    return mse


if __name__ == "__main__":
    # test for possion process:
    lambda_const = np.ones([20], dtype=float) * 4
    dt_data = np.ones([20, 1], dtype=float) / 4
    w_t = 0
    assert PP_mse(lambda_const, w_t, dt_data) < 1e-10, "possion precess failed"

    # test normal constraint:
    def integral(tao, lambda_0, w_t):
        if w_t != 0:
            res = lambda_0 * np.exp(w_t * tao) * np.exp(lambda_0 / w_t * (
            1 - np.exp(w_t * tao)))
            if np.isnan(res) or np.isinf(res):
                return 0
            return res
        else:
            res = lambda_0 * np.exp(-tao * lambda_0)
            if np.isnan(res) or np.isinf(res):
                return 0
            return res
    for _ in range(10):
        lambda_0 = np.random.uniform(0, 10)
        w_t = np.random.randint(0, 10)
        v, err = integrate.quad(integral, 0, np.inf, args=(lambda_0, w_t))
        assert abs(v - 1) < 1e-10, "normalization constraint failed"

    # test sense on w_t:
    last = 1.1
    for i in range(10):
        expected_d = np.sqrt(PP_mse([1.0], i, [[0]]))
        assert expected_d < last, "test on w_t failed"
        last = expected_d

    print "Test for pp_mse succeed!"


