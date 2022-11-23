"""
    Các công việc
    - Tìm hiểu cách dataloader hoạt động.
        + Input: images or text, cross_attention                          | x
        + Position Embeddings                                             | x
        + Masking and Paddings                                            | x

    - Tìm hiểu cách hoạt động của mô hình.                                | x
        + Encoder (Image Embeddings for Encoder)                          | x
        + Decoder                                                         | x
        + Improve Decoder                                                 | o
            - Greendy-Beam Search: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

    - Tìm hiểu cách tính độ chính xác và loss function                    | x
        + Đơn giản là category accuracy                                   | x



"""
import tensorflow as tf
from supervisor.tf_trainer import TFTrainer
from model.total_model import TotalModel
from utils.optimizer_helpers import CustomSchedule
from utils.tokens_helpers import get_vocab_from_huggingface
from DataLoader.dataloader_archiscribe import Dataset
from settings import config as cfg_training
from utils.metrics_helpers import softmax_ce_loss

if __name__ == '__main__':

    # get vocab information
    vocab_size = get_vocab_from_huggingface(name_model=cfg_training.MODEL_TOKENIZER)

    train_dataset = Dataset(record_path='./DatasetTFrecord/archiscribe-corpus/all_archiscribe.tfrec')
    train_dataset.load_tfrecord(repeat=True, batch_size=cfg_training.BATCH_SIZE,
                                with_padding_type="padding_have_right")

    architecture_model = TotalModel(
        enc_stack_size=cfg_training.ENC_STACK_SIZE,
        dec_stack_size=cfg_training.DEC_STACK_SIZE,
        num_heads=cfg_training.NUM_HEADS,
        d_model=cfg_training.D_MODEL,
        d_ff=cfg_training.D_FF,
        vocab_size=32000,
        max_seq_leng=cfg_training.MAX_LENGTH_SEQUENCE
    )

    architecture_model.compile()

    if cfg_training.LEARNING_RATE_TYPE == 'schedule':
        learning_rate = CustomSchedule(cfg_training.MODEL_SIZE)
    else:
        learning_rate = cfg_training.LEARCH_RATE

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    supervisor = TFTrainer(train_dataloader=train_dataset,
                           validation_dataloader=train_dataset,
                           model=architecture_model.model,
                           loss_fn=softmax_ce_loss,
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
