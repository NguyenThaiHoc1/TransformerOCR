import tensorflow as tf
from DataLoader.dataloader_archiscribe import Dataset


class BaseTrainer(object):

    def __init__(self, train_dataloader: Dataset,
                 validation_dataloader: Dataset,
                 model, loss_fn, optimizer,
                 save_freq, max_length_sequence,
                 monitor, mode, training_dir, name):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.save_freq = save_freq
        self.max_length_sequence = max_length_sequence
        self.monitor = monitor
        self.mode = mode
        self.training_dir = training_dir
        self.name = name

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def _train_step(self, datas):
        pass

    def _save_checkpoint(self):
        pass

    def restore(self, weights_only, from_scout):
        pass

    def export(self, model, export_dir):
        pass

    def train(self, epochs, steps_per_epoch):
        pass
