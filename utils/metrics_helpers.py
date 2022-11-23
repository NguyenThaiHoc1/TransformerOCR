import tensorflow as tf


def softmax_ce_loss(y_pred, y_true):
    y_true = tf.cast(y_true, 'int32')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
    loss = tf.keras.backend.mean(loss)
    return loss


def accuracy_on_max_tokens(y_pred, y_true):
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    corr = tf.keras.backend.cast(tf.keras.backend.equal(tf.keras.backend.cast(y_true, 'int32'),
                                                        tf.keras.backend.cast(tf.keras.backend.argmax(y_pred, axis=-1),
                                                                              'int32')), 'float32')
    corr = tf.keras.backend.sum(corr * mask, -1) / tf.keras.backend.sum(mask, -1)
    return tf.keras.backend.mean(corr)
