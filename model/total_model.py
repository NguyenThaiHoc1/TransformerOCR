import tensorflow as tf
from model.EmbeddingLayers.CustomEmbeddings import EmbeddingLayer
from model.Transformers.tf_trainsformer import Transformer


# tf.get_logger().setLevel('ERROR')


class TotalModel(tf.keras.Model):

    def __init__(self, name,
                 name_embedding_for_image,
                 input_shape, num_layers, d_model,
                 head_counts, dff,
                 input_vocab_size, target_vocab_size):
        """

        :param name: The name of architecture model
        :param input_shape: Input-shape which you will recivce from data
        :param num_layers: a number of layer of the BERT and GPT (Encoder and Decoder)
        :param d_model: Dimension of vector embeddings
        :param head_counts: the amount of heads of tranformer model
        :param dff: Deep Feed Foward
        :param input_vocab_size: Vocab-size of input-text
        :param target_vocab_size: Vocab-size of target-text (in OCR we dont need this properties)
        """
        super(TotalModel, self).__init__(name=name)
        # init hyper-parameter
        image_tensor = tf.keras.layers.Input(shape=input_shape, name="Image_tensor")

        # Embedding model for Image
        self.embedding_layers = EmbeddingLayer(base_model_name=name_embedding_for_image,
                                               vocab_size=input_vocab_size,
                                               d_model=d_model,
                                               name=f"{name_embedding_for_image}-Model")
        embedding_output_shape = self.embedding_layers.embedding_models(image_tensor)

        # Transformer model
        self.transformers = Transformer(num_layers=num_layers,
                                        d_model=d_model,
                                        head_counts=head_counts,
                                        dff=dff,
                                        maximum_position_encoding=embedding_output_shape.shape[1] *
                                                                  embedding_output_shape.shape[2],
                                        input_vocab_size=input_vocab_size,
                                        target_vocab_size=target_vocab_size,
                                        name="Transformer-Model")

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, *args, **kwargs):
        image, target = inputs
        training = kwargs.pop('training', False)

        enc_padding_mask = kwargs.pop('enc_padding_mask', None)
        look_ahead_mask = kwargs.pop('look_ahead_mask', None)
        dec_padding_mask = kwargs.pop('dec_padding_mask', None)

        image_embedding_patches = self.embedding_layers(image, training=training)
        out_embedding, attention_weights = self.transformers(inputs=(image_embedding_patches, target),
                                                             enc_padding_mask=enc_padding_mask,
                                                             look_ahead_mask=look_ahead_mask,
                                                             dec_padding_mask=dec_padding_mask,
                                                             training=training)

        out_embedding = self.final_layer(out_embedding)
        return out_embedding, attention_weights

    def summary(self, input_shape_of_image):
        image_tensor = tf.keras.layers.Input(shape=input_shape_of_image, name="Image_tensor")
        target = tf.keras.layers.Input(shape=(60,), name="Tokens_target")
        inputs = (image_tensor, target)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs, training=True), name=self.name)
        return model.summary()


if __name__ == '__main__':
    model = TotalModel(name="CheckingModel",
                       name_embedding_for_image="EmbeddingLayer",
                       input_shape=(150, 600, 3),
                       num_layers=5,
                       d_model=512,
                       head_counts=8,
                       dff=2048,
                       input_vocab_size=100,
                       target_vocab_size=100)
    model.summary(input_shape_of_image=(150, 600, 3))

    image_tensor = tf.random.uniform((10, 150, 600, 3))
    target = tf.random.uniform((10, 130))
    output_tensor, attention_weights = model((image_tensor, target), mask=None, training=False)
    print(f"{output_tensor.shape}")
    # model.save('./exported_model/')
