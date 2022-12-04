"""
https://stackoverflow.com/questions/60887949/keras-model-save-and-load-valueerror-could-not-find-matching-function-to-call
https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
"""
import tensorflow as tf
from model.EmbeddingLayers.CustomEmbeddings import EmbeddingLayer
from model.Transformers.subclass_transformer import Transformer


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

    def build_graph(self):
        image_inputs = tf.keras.layers.Input(shape=(32, 768, 3), dtype=tf.float32)
        tgt_seq_input = tf.keras.layers.Input(shape=(32,), dtype=tf.int32)
        image_enc_masks = tf.keras.layers.Input(shape=(1, 24), dtype=tf.float32)
        look_ahead_masks = tf.keras.layers.Input(shape=(32, 32), dtype=tf.int32)

        model = tf.keras.Model(inputs=[image_inputs, tgt_seq_input, image_enc_masks, look_ahead_masks],
                               outputs=self.call(image_inputs, tgt_seq_input, image_enc_masks, look_ahead_masks,
                                                 training=True))
        return model.summary()

    def call(self, image_inputs, tgt_seq_input=None, image_enc_masks=None, look_ahead_masks=None, training=None):
        images_emb = self.embeddings(image_inputs, training=training)
        final_output = self.transformer(images_emb, tgt_seq_input,
                                        enc_mask=image_enc_masks,
                                        look_ahead_mask=look_ahead_masks,
                                        training=training)
        final_output = self.final_layer(final_output)
        return final_output


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

    total_model.build_graph()

    dummy_batch_images = tf.random.uniform([3, 32, 768, 3], name='input_1')
    dummy_batch_targets = tf.random.uniform([3, 32], name='input_2')
    dummy_batch_enc_masks = tf.cast(tf.random.uniform([3, 1, 24]) < 0.5, dtype=tf.int32, name='input_3')
    dummy_batch_look_ahead_masks = tf.cast(tf.random.uniform([3, 32, 32]) < 0.5, dtype=tf.int32, name='input_4')

    out = total_model(dummy_batch_images,
                      dummy_batch_targets,
                      dummy_batch_enc_masks,
                      dummy_batch_look_ahead_masks, training=True)
    # total_model.save("../exported_model_test")

    print(f"Output shape: {out.shape}")
