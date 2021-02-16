import tensorflow as tf


def softmax_cross_entropy_with_logits_loss(
        labels=None, output=None, num_labels=None):
    if num_labels is not None and num_labels == 1:
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=output))
    else:
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=output))
    return loss


def log_prob_loss(labels=None, output=None, num_labels=None):
    output = tf.nn.sigmoid(output)
    loss = tf.negative(tf.reduce_mean(
        tf.multiply(labels, tf.log(output))) +
                       tf.multiply(1 - labels, tf.log(1 - output)))
    return loss


def tanh_l2_norm_loss(labels=None, output=None, num_labels=None):
    return tf.reduce_mean(tf.square(labels - tf.tanh(output)))


def l2_norm_loss(labels=None, output=None, num_labels=None):
    return tf.reduce_mean(tf.square(labels - output))


def weighted_l2(**kwargs):
    try:
        return tf.losses.mean_squared_error(
            labels=kwargs["labels"],
            predictions=kwargs["output"],
            weights=kwargs["mask"])
    except KeyError:
        return tf.losses.mean_squared_error(
            labels=kwargs["labels"],
            predictions=kwargs["output"])


def pp_likelihood(markers=None, output=None, **kwargs):
    """
    :param markers: one-hot markers, indicating type of events
    :param output: a tuple, (P(y|h), f(t))
    :param kwargs: not used
    :return: loss(i.e. \sum_i \sum_j (log(P(y|h) + log(f(t))))
    """
    multinomial, likelihood = output
    markers = tf.reshape(markers, [-1, markers.shape.as_list()[-1]])
    gather_idx = tf.where(markers > 0)
    overflow_constant = tf.constant(1e-32, dtype=tf.float32)
    log_p = tf.log(
        tf.maximum(tf.gather_nd(multinomial, gather_idx),
               overflow_constant)
        )
    log_f = tf.log(tf.maximum(likelihood, overflow_constant))
    likelihood = tf.add(tf.reduce_mean(log_p), tf.reduce_mean(log_f))
    return tf.negative(likelihood)


if __name__ == "__main__":
    sess = tf.Session()
    import numpy as np
    output = np.random.uniform(-50, 50, [2, 3])
    labels = np.random.multinomial(1, [0.33, 0.33, 0.34], 2).astype(float)
    print output
    print labels
    loss = log_prob_loss(labels=labels, output=output)
    print sess.run(loss)
