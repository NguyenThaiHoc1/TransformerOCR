"""
    https://www.tensorflow.org/text/tutorials/transformer
    https://keras.io/guides/understanding_masking_and_padding/
    https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer
    https://github.com/Lornatang/TensorFlow2-tutorials/blob/master/Experts_tutorial/Text/transformer.py
    https://medium.com/geekculture/scene-text-recognition-using-resnet-and-transformer-c1f2dd0e69ae
"""
import tensorflow as tf
import numpy as np


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) # cai nay do the dung pretrain
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, training=False, mask=None):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class ScaleDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, head_dimension, name):
        super(ScaleDotProductAttention, self).__init__(name=name)
        self.head_dimension = head_dimension

    def call(self, inputs, mask=None, training=False):
        """Calculate the attention weights.
          q, k, v must have matching leading dimensions.
          k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
          The mask has different shapes depending on its type(padding or look ahead)
          but it must be broadcastable for addition.

          Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable
                  to (..., seq_len_q, seq_len_k). Defaults to None.

          Returns:
            output, attention_weights
        """
        query, key, value = inputs
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(self.head_dimension, tf.float32)  # depth
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class MultiHeadAttentionCustom(tf.keras.layers.Layer):

    def __init__(self, head_count, d_model, name):
        super(MultiHeadAttentionCustom, self).__init__(name=name)
        # init some attribute
        self.attention_head_count = head_count
        self.d_model = d_model
        self.heads_dimension = d_model // head_count

        # create weight of keys, values, querys
        self.w_key = tf.keras.layers.Dense(d_model)
        self.w_value = tf.keras.layers.Dense(d_model)
        self.w_query = tf.keras.layers.Dense(d_model)

        # scale dot-product attention
        self.scale_dot_attention = ScaleDotProductAttention(self.heads_dimension,
                                                            name=f"{name}_ScaleDotProductAttention")

        # linear
        self.feedfoward = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        return tf.transpose(
            tf.reshape(
                x, shape=[batch_size, -1, self.attention_head_count, self.heads_dimension]
            ),  # have shape = [batch_size, tokens | seq_len, num_head, dimension of head]
            perm=[0, 2, 1, 3]  # shape = [batch_size, num_head, tokens | seq_len, dimension of head]
        )

    def concat_heads(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(
                tensor,
                perm=[0, 2, 1, 3]
            ),  # shape = [batch_size, tokens | seq_len, num_head, dimension of head]
            shape=[batch_size, -1, self.d_model]  # shape = [batch_size, seq_len, num_head * dimension of head]
        )

    def call(self, inputs, mask=None, training=False):
        query, key, value = inputs
        # input: batch x tokens x d_model
        batch_size = tf.shape(query)[0]

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        output, attention = self.scale_dot_attention((query, key, value), mask)
        output = self.concat_heads(output, batch_size)

        return self.feedfoward(output), attention


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, mask=None, training=False):
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x, training=training)
        return x


class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, head_count, d_model, name):
        super().__init__(name=name)
        self.multi_head_attention = MultiHeadAttentionCustom(head_count, d_model, name=f"{name}_MultiHeadAttention")
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):

    def __init__(self, head_count, d_model, name):
        super(CrossAttention, self).__init__(head_count, d_model, name=name)

    def call(self, inputs, mask=None, training=False):
        x, context = inputs
        # query, key, value
        attn_output, attention = self.multi_head_attention(
            (x, context, context),
            mask=mask,
            training=training
        )

        out = self.add([x, attn_output])
        out = self.layernorm(out, training=training)
        return out, attention


class GlobalSelfAttention(BaseAttention):

    def __init__(self, head_count, d_model, name):
        super(GlobalSelfAttention, self).__init__(head_count, d_model, name=name)

    def call(self, x, mask=None, training=False):
        attn_output, attention = self.multi_head_attention(
            (x, x, x),
            mask=mask,
            training=training
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x, training=training)
        return x, attention


class CausalSelfAttention(BaseAttention):

    def __init__(self, head_count, d_model, name):
        super(CausalSelfAttention, self).__init__(head_count, d_model, name=name)

    def call(self, x, mask=None, training=False):
        attn_output, attention = self.multi_head_attention(
            (x, x, x),
            mask=mask,
            training=training
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x, training=training)
        return x, attention


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, head_count, d_model, dff, name, dropout_rate=0.1):
        super().__init__(name=name)
        self.global_self_attention = GlobalSelfAttention(head_count, d_model, name=f"{name}_GlobalSelfAttention")

        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x, mask=None, training=False):
        out, attention = self.global_self_attention(x, training=training)
        out = self.ffn(out, training=training)
        return out, attention


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, head_count, df, maximum_position_encoding, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.head_count = head_count
        self.dff_vocab_size = df
        self.dropout_rate = dropout_rate
        self.maximum_position_encoding = maximum_position_encoding

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)

        self.enc_layer = [
            EncoderLayer(
                head_count=head_count, d_model=d_model,
                dff=df,
                dropout_rate=dropout_rate,
                name=f"EncoderLayer-{i}"
            ) for i in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[tf.newaxis, :seq_len, :]

        for i in range(self.num_layers):
            x, attention = self.enc_layer[i](x, mask=mask, training=training)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, head_count, d_model, dff, name, dropout_rate=0.1):
        super().__init__(name=name)
        self.causal_self_attention = CausalSelfAttention(head_count, d_model, name=f"{name}_CausalSelfAttention")
        self.cross_attention = CrossAttention(head_count, d_model, name=f"{name}_CrossAttention")
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, inputs, look_ahead_mask=None, padding_mask=None, training=False):
        x, context = inputs
        x, attention_1 = self.causal_self_attention(x=x, mask=look_ahead_mask, training=training)
        x, attention_2 = self.cross_attention(inputs=(x, context), mask=padding_mask, training=training)
        x = self.ffn(x, training=training)  # Shape `(batch_size, seq_len, d_model)`.
        return x, attention_1, attention_2


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, head_count, df, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.head_count = head_count
        self.dff_vocab_size = df
        self.dropout_rate = dropout_rate

        self.tokens_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.dec_layers = [
            DecoderLayer(
                head_count=head_count,
                d_model=d_model,
                dff=df,
                name=f"DecoderLayer-{i}",
                dropout_rate=dropout_rate
            ) for i in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, look_ahead_mask=None, padding_mask=None, training=False):
        x, context = inputs
        attention_weights = {}

        x = self.tokens_embedding(x)
        # Add dropout.
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention_1, attention_2 = self.dec_layers[i]((x, context), look_ahead_mask, padding_mask, training=training)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = attention_1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = attention_2
        return x, attention_weights  # The shape of x is (batch_size, target_seq_len, d_model).


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, head_counts, dff,
                 maximum_position_encoding,
                 input_vocab_size, target_vocab_size, name, dropout_rate=0.1):
        super().__init__(name=name)
        self.encoder = Encoder(
            num_layers,
            d_model,
            head_counts,
            dff,
            maximum_position_encoding,
            input_vocab_size,
            dropout_rate=dropout_rate
        )  # BERT

        self.decoder = Decoder(
            num_layers,
            d_model,
            head_counts,
            dff,
            target_vocab_size,
            dropout_rate=dropout_rate
        )  # GPT 3

    def call(self, inputs, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None, training=False):
        x, target = inputs
        context = self.encoder(x, training=training, mask=enc_padding_mask)  # (batch_size, context_len, d_model)
        x, attention_weights = self.decoder((target, context), look_ahead_mask, dec_padding_mask,
                                            training=training)  # (batch_size, target_len, d_model)
        return x, attention_weights


if __name__ == '__main__':
    sample_ca = Transformer(num_layers=5,
                            d_model=512,
                            head_counts=8,
                            dff=2048,
                            maximum_position_encoding=1192,
                            input_vocab_size=32000,
                            target_vocab_size=32000,
                            name="Transformer-Model")
    image_embedding = tf.random.uniform((1, 60, 512))
    target_embedding = tf.random.uniform((1, 32))
    context = None
    out, attention_weights = sample_ca(
        inputs=(image_embedding, target_embedding)
    )
    print(out.shape)
    print(attention_weights)
    # sample_ca.save('../../exported_model')
