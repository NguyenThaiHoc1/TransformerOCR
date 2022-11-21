import tensorflow as tf
from supervisor.tf_trainer import TFTrainer
from model.total_model_v2 import TotalModel
from utils.optimizer_helpers import CustomSchedule
from utils.tokens_helpers import get_vocab_from_huggingface
from DataLoader.dataloader_archiscribe import Dataset
from settings import config as cfg_training

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':

    # get vocab information
    vocab_size = get_vocab_from_huggingface(name_model=cfg_training.MODEL_TOKENIZER)

    train_dataset = Dataset(record_path='./DatasetTFrecord/archiscribe-corpus/all_archiscribe.tfrec')
    train_dataset.load_tfrecord(repeat=True, batch_size=cfg_training.BATCH_SIZE,
                                with_padding_type="padding_have_right")

    model = TotalModel(
        enc_stack_size=5,
        dec_stack_size=5,
        num_heads=4,
        d_model=512,
        d_ff=2048,
        vocab_size=32000,
        max_seq_leng=70
    )

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
