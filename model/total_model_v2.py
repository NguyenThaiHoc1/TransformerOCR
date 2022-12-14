import tensorflow as tf
import numpy as np
from model.EmbeddingLayers.CustomEmbeddings import EmbeddingLayer
from model.Transformers.transformer_test import Transformer


class TotalModel(tf.keras.Model):
    def __init__(self, enc_stack_size, dec_stack_size,
                 num_heads, d_model, d_ff,
                 vocab_size,
                 max_seq_leng):
        super(TotalModel, self).__init__()
        # init param
        self.enc_stack_size = enc_stack_size
        self.dec_stack_size = dec_stack_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_leng = max_seq_leng
        input_shape = (300, 1200, 3)

        image_tensor = tf.keras.layers.Input(shape=input_shape, name="Image_tensor")

        self.embeddings = EmbeddingLayer(base_model_name="InceptionV3",
                                         d_model=self.d_model,
                                         vocab_size=self.vocab_size,
                                         name="InceptionV3_Model")

        embedding_output_shape = self.embeddings.embedding_models(image_tensor)

        self.transformer = Transformer(enc_stack_size, dec_stack_size, num_heads, d_model, d_ff,
                                       vocab_size, max_seq_leng=max_seq_leng,
                                       embedding_shape=embedding_output_shape.shape[1] * embedding_output_shape.shape[2],
                                       dropout_rate=0.1)

    def call(self, image_inputs, tgt_seq_input=None, image_enc_masks=None, look_ahead_masks=None, training=None):
        img_embeddings = self.embeddings(image_inputs)
        out, _ = self.transformer(img_embeddings, tgt_seq_input,
                                  training=True,
                                  look_ahead_mask=look_ahead_masks,
                                  dec_padding_mask=image_enc_masks,
                                  enc_padding_mask=image_enc_masks)

        return out


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


if __name__ == '__main__':
    total_model = TotalModel(
        enc_stack_size=5,
        dec_stack_size=5,
        num_heads=4,
        d_model=512,
        d_ff=2048,
        vocab_size=32000,
        max_seq_leng=70
    )
    # 32, 768
    dummy_batch_images = tf.random.uniform([3, 300, 1200, 3], name='input_1')
    dummy_batch_targets = tf.random.uniform([3, 70], name='input_2')

    # dummy_batch_look_ahead_masks = create_masks_decoder(dummy_batch_targets)
    # dummy_batch_enc_masks = tf.cast(tf.random.uniform([3, 1, 24]) < 0.5, dtype=tf.int32, name='input_3')
    dummy_batch_look_ahead_masks = tf.cast(tf.random.uniform([3, 70, 70]) < 0.5, dtype=tf.int32, name='input_4')

    look_ahead_mask = create_look_ahead_mask(70)

    out = total_model(dummy_batch_images,
                      dummy_batch_targets,
                      None,
                      look_ahead_masks=look_ahead_mask,
                      training=True)
    # total_model.save("../exported_model_test")

    print(f"Output shape: {out.shape}")
