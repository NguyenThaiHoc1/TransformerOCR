import tensorflow as tf
from matplotlib import pyplot as plt
from supervisor.tf_trainer import TFTrainer
from model.total_model import TotalModel
from utils.optimizer_helpers import CustomSchedule
from utils.tokens_helpers import get_vocab_from_huggingface
from DataLoader.dataloader_archiscribe import Dataset
from settings import config as cfg_training

if __name__ == '__main__':

    # get vocab information
    vocab_size = get_vocab_from_huggingface(name_model=cfg_training.MODEL_TOKENIZER)

    train_dataset = Dataset(record_path='./DatasetTFrecord/archiscribe-corpus/all_archiscribe.tfrec')
    train_dataset.load_tfrecord(repeat=False, batch_size=cfg_training.BATCH_SIZE)

    model = TotalModel(name="AttentionOCR-Model",
                       name_embedding_for_image=cfg_training.IMAGES_EMBEDDING_TYPE,
                       input_shape=cfg_training.INPUT_SHAPE,
                       num_layers=5,
                       d_model=cfg_training.MODEL_SIZE,
                       head_counts=8,
                       dff=2048,
                       input_vocab_size=vocab_size,
                       target_vocab_size=vocab_size)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if cfg_training.LEARNING_RATE_TYPE == 'schedule':
        learning_rate = CustomSchedule(cfg_training.MODEL_SIZE)
    else:
        learning_rate = cfg_training.LEARCH_RATE

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    supervisor = TFTrainer(train_dataloader=train_dataset,
                           validation_dataloader=train_dataset,
                           model=model,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           save_freq=cfg_training.SAVE_FREQ,
                           max_length_sequence=cfg_training.MAX_LENGTH_SEQUENCE,
                           monitor="loss",
                           mode="min",
                           training_dir="./logs_training",
                           name="Trainer_Supervisor")
    supervisor.restore(weights_only=False, from_scout=True)
    supervisor.train(epochs=cfg_training.EPOCHS,
                     steps_per_epoch=cfg_training.NUM_SAMPLES // cfg_training.BATCH_SIZE)
    supervisor.export(export_dir=cfg_training.EXPORT_DIR)
