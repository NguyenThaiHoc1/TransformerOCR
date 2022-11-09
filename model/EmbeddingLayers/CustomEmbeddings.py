"""
    Đặt cờ trong training: https://github.com/tensorflow/tensorflow/issues/36936
    Block dropout: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
    training vs trainable:https://stackoverflow.com/questions/50209310/significance-of-trainable-and-training-flag-in-tf-layers-batch-normalization
"""
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, base_model_name, name, **kwargs):
        super(EmbeddingLayer, self).__init__(name=name)
        self.base_model_name = base_model_name
        end_point = None
        if base_model_name == 'EfficientNetB0':
            base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]

        elif base_model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]

        elif base_model_name == 'InceptionResNetV2':
            end_point = 'mixed_6a'
            base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        else:
            base_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(150, 600, 3)),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

                tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

                tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

                tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

                tf.keras.layers.Conv2D(512, (2, 2), padding='valid', activation='relu',
                                       kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(rate=0.5)
            ])
            base_model_layers = [layer.name for layer in base_model.layers]

        if end_point is not None:
            assert end_point in base_model_layers, "no {} layer in {}".format(end_point, base_model_name)
            base_model_input = base_model.input
            base_model_output = base_model.get_layer(name=end_point).output
        else:
            base_model_input = base_model.input
            base_model_output = base_model.output

        self.embedding_models = tf.keras.Model(inputs=base_model_input, outputs=base_model_output, name=base_model_name)
        # self.conv_out_shape = self.conv_model.predict(np.array([np.zeros(hparams.image_shape)])).shape

    def call(self, inputs, *args, **kwargs):
        training = kwargs.pop('training', False)
        out = self.embedding_models(inputs, training=training)

        batch_size = tf.shape(inputs)[0]
        feature_size = tf.shape(out)[-1]
        out = tf.reshape(out, shape=[batch_size, -1, feature_size], name='ReshapeEmbeddingLayer')
        return out


class ModelEmbedding(tf.keras.Model):

    def __init__(self):
        super(ModelEmbedding, self).__init__()
        self.embeddings = EmbeddingLayer(base_model_name="EmbeddingLayer",
                                         d_model=512,
                                         vocab_size=32000,
                                         name="Hello")

    def call(self, inputs, training=None, mask=None):
        out = self.embeddings(inputs, training)
        return out


if __name__ == '__main__':
    model = ModelEmbedding()
    # model.build(input_shape=(20, 150, 600, 3))
    # model.summary()
    input_tensor = tf.random.uniform((10, 150, 600, 3))
    output_tensor = model(input_tensor)
    print(f"{output_tensor.shape}")

    # model.save('./exported_model/')
