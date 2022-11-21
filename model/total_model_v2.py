import tensorflow as tf
from model.EmbeddingLayers.CustomEmbeddings import EmbeddingLayer
from model.Transformers.tf_next_model import Transformer


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
            max_seq_leng=self.max_seq_leng
        )

    def __call__(self, images, targets, enc_masks, look_ahead_masks, training=False):
        images_embs = self.embeddings(images)
        output = self.transformer(images_embs, targets,
                                  enc_mask=enc_masks, look_ahead_mask=look_ahead_masks,
                                  training=training)
        return output


if __name__ == '__main__':
    """
        TEST TOTAL MODEL
        NOTE: tại sao lại là 24 vì
        768 = 3 * 8 * 32 (mà resnet-18 = 768 / 32 = 24) tuy vậy 
        =khi đi qua resnet-18=> SHAPE: (b, 32/32, 768/32, C or d_model) ==> MASK_SHAPE (b, 1, 768/32)
    """
    # create inputs
    dummy_batch_images = tf.random.uniform([3, 32, 768, 3])
    dummy_batch_targets = tf.random.uniform([3, 37])

    dummy_batch_enc_masks = tf.cast(tf.random.uniform([3, 1, 24]) < 0.5, dtype=tf.float32)
    dummy_batch_look_ahead_masks = tf.cast(tf.random.uniform([3, 37, 37]) < 0.5, dtype=tf.float32)

    # create model
    model = TotalModel(
        enc_stack_size=5,
        dec_stack_size=5,
        num_heads=4,
        d_model=512,
        d_ff=2048,
        vocab_size=32000,
        max_seq_leng=70
    )

    out = model(
        images=dummy_batch_images,
        targets=dummy_batch_targets,
        enc_masks=dummy_batch_enc_masks,
        look_ahead_masks=dummy_batch_look_ahead_masks,
        training=True
    )

    print(f"TOTAL-MODEL: {out.shape}")
    # END
