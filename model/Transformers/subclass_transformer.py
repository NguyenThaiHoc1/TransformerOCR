"""
    Base-on: https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
    https://www.graviti.com/article/guide-to-ocr-transformer#datas

    *NOTE: Khi viết model bằng tensorflow.keras.layers
    Để Debugging mô hình
    ta không nên thực hiện việc kế thứa các class "tf.keras.layers.Layer" or "tf.keras.Model"
    Hãy viết dạng form dưới đây

    class TestLayer:
        def __init__(self):
            pass

        def __call__(self, inputs, trainings=False):
            pass

    như vậy sẽ dễ debug hơn
    Khi hoàn thành thì hẵng thêm các kế thừa từ "tf.keras.layers.Layer" ("Không ảnh hương kết quả")
"""
import tensorflow as tf
import numpy as np


def getpositionencodingmask(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class PosEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, d_emb):
        super(PosEncodingLayer, self).__init__()
        self.pos_emb_matrix = tf.keras.layers.Embedding(max_len, d_emb, trainable=False,
                                                        weights=[getpositionencodingmask(max_len, d_emb)])

    def get_pos_seq(self, x):
        mask = tf.keras.backend.cast(tf.keras.backend.not_equal(x, 0), 'int32')
        pos = tf.keras.backend.cumsum(tf.keras.backend.ones_like(x, 'int32'), 1)
        return pos * mask

    def call(self, seq, pos_input=False, training=False):
        x = seq
        if not pos_input:
            x = tf.keras.layers.Lambda(self.get_pos_seq)(x)
        return self.pos_emb_matrix(x)


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)  # cai nay do the dung pretrain

    def call(self, inputs, mask=None, training=False):
        output = self.embedding(inputs)
        return output * tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = tf.keras.layers.Dropout(attn_dropout)

    def call(self, q, k, v, mask=None, training=False):  # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1], axes=[2, 2]) / x[2])(
            [q, k, temper])  # shape=(batch, q, k)
        if mask is not None:
            mmask = tf.keras.layers.Lambda(lambda x: (-1e+9) * (1. - tf.keras.backend.cast(x, 'float32')))(mask)
            attn = tf.keras.layers.Add()([attn, mmask])
        attn = tf.keras.layers.Activation('softmax')(attn)
        attn = self.dropout(attn, training)
        output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadedAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "Check MultiHeadedAttention"

        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # create weight of keys, values, querys
        self.w_key = tf.keras.layers.Dense(self.d_k * self.num_heads)
        self.w_value = tf.keras.layers.Dense(self.d_k * self.num_heads)
        self.w_query = tf.keras.layers.Dense(self.d_k * self.num_heads)

        self.attention = ScaledDotProductAttention()

        self.feedfoward = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None, training=False):
        """

        :param query: SHAPE - [b, sequence_len, C or d_model]
        :param key: SHAPE - [b, sequence_len, C or d_model]
        :param value: SHAPE - [b, sequence_len, C or d_model]
        :param mask: SHAPE - [b, 1, sequence_len]
        :param training: Boolean
        :return:
        """
        n_head = self.num_heads
        d_v, d_k = self.d_v, self.d_k

        # [b, sequence_len, n_head * d_k]
        qs = self.w_query(query)  # [batch_size, len_seq, n_head * d_k]
        ks = self.w_key(key)
        vs = self.w_value(value)

        def reshape1(x):
            s = tf.shape(x)  # [batch_size, len_seq, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])  # [b, len_seq, n_head, (n_head * d_k) // n_head]
            x = tf.transpose(x, [2, 0, 1, 3])  # [n_head, b, len_seq, d_k]
            x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_seq, d_k]
            return x

        def reshape2(x):
            s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [n_head, -1, s[1], s[2]])
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
            return x

        qs = tf.keras.layers.Lambda(reshape1)(qs)
        ks = tf.keras.layers.Lambda(reshape1)(ks)
        vs = tf.keras.layers.Lambda(reshape1)(vs)

        if mask is not None:
            mask = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, n_head, 0))(mask)

        head, attn = self.attention(qs, ks, vs, mask=mask, training=training)
        head = tf.keras.layers.Lambda(reshape2)(head)
        outputs = self.feedfoward(head)
        outputs = tf.keras.layers.Dropout(self.dropout_rate)(outputs, training)
        return outputs, attn


class PositionwiseFeedForward(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.backbone_feature = tf.keras.Sequential([
            tf.keras.layers.Conv1D(d_ff, 1, activation='relu'),
            tf.keras.layers.Conv1D(d_model, 1)
        ])
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        output = self.backbone_feature(inputs, training)
        output = self.dropout(output, training)
        output = tf.keras.layers.Add()([output, inputs])
        output = self.layer_norm(output)
        return output


class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model):
        super(BaseAttention, self).__init__()
        self.mha = MultiHeadedAttention(num_heads=num_heads, d_model=d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):

    def call(self, query, value, key, mask=None, training=False):
        output, slf_attn = self.mha(query, value, key, mask, training)
        output = self.add([query, output])
        output = self.layer_norm(output)
        return output, slf_attn


class CausalSelfAttention(BaseAttention):

    def call(self, query, value, key, mask=None, training=False):
        output, slf_attn = self.mha(query, value, key, mask, training)
        output = self.add([query, output])
        output = self.layer_norm(output)
        return output, slf_attn


class CrossAttention(BaseAttention):

    def call(self, query, value, key, mask=None, training=False):
        output, slf_attn = self.mha(query, value, key, mask, training)
        output = self.add([query, output])
        output = self.layer_norm(output)
        return output, slf_attn


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff, name):
        super(EncoderLayer, self).__init__()
        self.attention_layer = GlobalSelfAttention(num_heads=num_heads, d_model=d_model)
        self.positionwise_feedfoward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    def call(self, enc_inputs, mask=None, training=False):
        output, slf_attn = self.attention_layer(
            enc_inputs, enc_inputs, enc_inputs,
            mask,
            training
        )
        output = self.positionwise_feedfoward(output, training)
        return output, slf_attn


class Encoder(tf.keras.layers.Layer):

    def __init__(self, stack_size, num_heads, d_model, d_ff, name):
        super(Encoder, self).__init__()
        self.stack_size = stack_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.enc_layers = [
            EncoderLayer(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                name=f"EncoderLayer{i}"
            ) for i in range(stack_size)
        ]

    def call(self, enc_inputs, return_att=True, mask=None, training=False):
        output = enc_inputs
        if return_att:
            atts = []
        for i in range(self.stack_size):
            output, att = self.enc_layers[i](output, mask=mask, training=training)
            if return_att:
                atts.append(att)

            #############################
            # Có nên thêm layer-norm ở khúc này ????
            #############################
        return (output, atts) if return_att else output


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff, name):
        super(DecoderLayer, self).__init__()
        self.attention_layer = CausalSelfAttention(num_heads=num_heads, d_model=d_model)
        self.cross_attention_layer = CrossAttention(num_heads=num_heads, d_model=d_model)
        self.positionwise_feedfoward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    def call(self, decoder_inputs, encoder_inputs, look_ahead_mask=None, decoder_mask=None, training=False):
        output, slf_attn_1 = self.attention_layer(
            decoder_inputs, decoder_inputs, decoder_inputs,
            mask=look_ahead_mask,
            training=training
        )
        output, slf_attn_2 = self.cross_attention_layer(
            output, encoder_inputs, encoder_inputs,
            mask=decoder_mask,  # ở 1 số bài viết thì deocder_mask = encoder_mask
            training=training
        )
        output = self.positionwise_feedfoward(output, training=training)
        return output, slf_attn_1, slf_attn_2


class Decoder(tf.keras.layers.Layer):

    def __init__(self, stack_size, num_heads, d_model, d_ff, vocab_size, max_seq_leng, name):
        super(Decoder, self).__init__()
        self.stack_size = stack_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_leng = max_seq_leng

        self.tokens_embedding = Embeddings(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = PosEncodingLayer(max_seq_leng, d_model)

        self.dec_layers = [
            DecoderLayer(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                name=f"DecoderLayer{i}"
            ) for i in range(stack_size)
        ]

    def call(self, target, enc_outputs, return_att=True, look_ahead_mask=None, mask=None, training=False):
        target_emb = self.tokens_embedding(target)
        target_emb = tf.keras.layers.Lambda(lambda x: x[0] + x[1],
                                            output_shape=lambda x: x[0])([target_emb, self.pos_emb(target)])

        if return_att:
            self_atts, enc_atts = [], []

        for i in range(self.stack_size):
            output, self_att, enc_att = self.dec_layers[i](target_emb, enc_outputs,
                                                           look_ahead_mask=look_ahead_mask,
                                                           decoder_mask=mask, training=training)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        return (output, self_atts, enc_atts) if return_att else output


class Transformer(tf.keras.Model):

    def __init__(self, enc_stack_size, dec_stack_size, num_heads, d_model, d_ff, vocab_size, max_seq_leng, name):
        super(Transformer, self).__init__()
        self.enc_layers = Encoder(
            stack_size=enc_stack_size,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            name="Encoder"
        )  # BERT - for IMAGE

        self.dec_layers = Decoder(
            stack_size=dec_stack_size,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_leng=max_seq_leng,
            name="Decoder"
        )  # GPT.3 - for WORDS

    def call(self, images_batch, target, enc_mask=None, look_ahead_mask=None, training=False):
        enc_outputs, _ = self.enc_layers(images_batch, mask=enc_mask, training=training)
        dec_outputs, _, _ = self.dec_layers(target, enc_outputs,
                                            look_ahead_mask=look_ahead_mask,
                                            mask=enc_mask,
                                            training=training)
        return dec_outputs


if __name__ == '__main__':
    """
        TEST OF EACH COMPONENTS
    """
    enc_inputs = query = key = value = tf.random.uniform((3, 24, 512))
    encoder_padding_mask = tf.random.uniform((3, 1, 24))

    model = Encoder(
        stack_size=5,
        num_heads=4,
        d_model=512,
        d_ff=2048,
        name="Encoder"
    )

    out, _ = model(
        enc_inputs=enc_inputs,
        mask=encoder_padding_mask,
        training=False
    )

    print(f"Encoder Output_shape: {out.shape}")

    target_inputs = tf.random.uniform((3, 80))
    look_ahead_mask = tf.random.uniform((3, 80, 80))

    model = Decoder(
        stack_size=5,
        num_heads=4,
        d_model=512,
        d_ff=2048,
        vocab_size=100,
        max_seq_leng=100,
        name="Decoder"
    )

    out, _, _ = model(
        target=target_inputs,
        enc_outputs=out,
        return_att=True,
        look_ahead_mask=look_ahead_mask,
        mask=encoder_padding_mask,
        training=True
    )

    print(f"Decoder Output_shape: {out.shape}")

    """
        TEST OF TRANSFORMER MODEL
    """

    model = Transformer(
        enc_stack_size=5,
        dec_stack_size=5,
        num_heads=4,
        d_model=512,
        d_ff=2048,
        vocab_size=32000,
        max_seq_leng=100,
        name="TransformersModel"
    )

    out = model(
        enc_inputs,
        target_inputs,
        encoder_padding_mask,
        look_ahead_mask,
        training=True
    )

    print(f"Transformer-Model Output_shape: {out.shape}")
