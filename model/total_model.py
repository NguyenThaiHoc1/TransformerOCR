import tensorflow as tf
from model.Transformers.transformer import Transformer
from model.EmbeddingLayers.CustomEmbeddings import EmbeddingLayer


class TotalModel:

    def __init__(self, enc_stack_size, dec_stack_size,
                 num_heads, d_model, d_ff,
                 vocab_size,
                 max_seq_leng):
        # init param
        self.enc_stack_size = enc_stack_size
        self.dec_stack_size = dec_stack_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_leng = max_seq_leng

        self.embeddings = EmbeddingLayer(base_model_name="ResNet-18",
                                         d_model=self.d_model,
                                         vocab_size=self.vocab_size,
                                         name="ResNet18_Model")

        self.transformer = Transformer(
            enc_stack_size=self.enc_stack_size,
            dec_stack_size=self.dec_stack_size,
            num_heads=self.num_heads,
            d_model=self.d_model,
            d_ff=self.d_ff,
            vocab_size=self.vocab_size,
            max_seq_leng=self.max_seq_leng,
            name="Transformer_Model"
        )

        self.final_layer = tf.keras.layers.Dense(self.vocab_size, use_bias=False)

        # init model
        self.model = None

    def compile(self):
        image_inputs = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.float32)
        image_enc_masks = tf.keras.layers.Input(shape=(1, None), dtype=tf.float32)
        tgt_seq_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        look_ahead_masks = tf.keras.layers.Input(shape=(None, None), dtype=tf.int32)

        images_emb = self.embeddings(image_inputs, training=False)
        final_output = self.transformer(images_emb, tgt_seq_input,
                                        enc_mask=image_enc_masks,
                                        look_ahead_mask=look_ahead_masks,
                                        training=True)
        final_output = self.final_layer(final_output)

        self.model = tf.keras.Model([image_inputs, image_enc_masks, tgt_seq_input, look_ahead_masks], final_output)


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

    total_model.compile()

    total_model.model.summary()
