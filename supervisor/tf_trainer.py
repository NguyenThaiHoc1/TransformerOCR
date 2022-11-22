import os
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import tensorflow as tf
from supervisor.trainer import BaseTrainer
from utils.tensorflow_helpers import create_look_ahead_mask, create_padding_mask


def get_accu(y_pred, y_true):
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    corr = tf.keras.backend.cast(tf.keras.backend.equal(tf.keras.backend.cast(y_true, 'int32'),
                                                        tf.keras.backend.cast(tf.keras.backend.argmax(y_pred, axis=-1),
                                                                              'int32')), 'float32')
    corr = tf.keras.backend.sum(corr * mask, -1) / tf.keras.backend.sum(mask, -1)
    return tf.keras.backend.mean(corr)


class TFTrainer(BaseTrainer):

    def __init__(self, train_dataloader,
                 validation_dataloader,
                 model, loss_fn, optimizer,
                 save_freq, max_length_sequence,
                 monitor, mode, training_dir, name):
        super(TFTrainer, self).__init__(train_dataloader,
                                        validation_dataloader,
                                        model, loss_fn, optimizer,
                                        save_freq, max_length_sequence,
                                        monitor, mode, training_dir, name)

        self.metrics = {
            'loss': tf.keras.metrics.Mean(name="train_loss_mean", dtype=tf.float32),
        }

        self.epoch_metrics = {
            'loss': tf.keras.metrics.Mean(name="train_epoch_loss_mean", dtype=tf.float32),
        }

        self.monitor = self.metrics[self.monitor]

        self.schedule = {
            'step': tf.Variable(0, trainable=False, dtype=tf.int64),
            'epoch': tf.Variable(1, trainable=False, dtype=tf.int64),
            'monitor_value': tf.Variable(0, trainable=False, dtype=tf.float32)
        }

        # checkpoint-scout
        self.checkpoint_scout = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            metrics=self.metrics,
            schedule=self.schedule,
            monitor=self.monitor,
        )

        # checkpoint-manager
        self.checkpoint_manager = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            metrics=self.metrics,
            schedule=self.schedule,
            monitor=self.monitor,
        )

        # A model manager is responsible for saving the current training
        # schedule and the model weights.
        self.manager = tf.train.CheckpointManager(
            self.checkpoint_manager,
            os.path.join(training_dir, "checkpoints"),
            max_to_keep=5
        )

        # A model scout watches and saves the best model according to the
        # monitor value.
        self.scout = tf.train.CheckpointManager(
            self.checkpoint_scout,
            os.path.join(training_dir, 'model_scout'),
            max_to_keep=1
        )

        # A clerk writes the training logs to the TensorBoard.
        dt_string = datetime.now().strftime("%d%m%Y_%H_%M_%S")
        self.clerk = tf.summary.create_file_writer(os.path.join(training_dir, 'logs', dt_string))

    def _train_step(self, data):
        batch_images, batch_targets, encoder_masks, look_ahead_masks = data
        encoder_masks = tf.cast(encoder_masks, dtype=tf.float32)
        look_ahead_masks = tf.cast(look_ahead_masks, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predictions = self.model([batch_images, encoder_masks, batch_targets, look_ahead_masks], training=True)
            loss = self.loss_fn(predictions, batch_targets)
            acc = get_accu(predictions, batch_targets)
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss, acc

    def _reset_metrics(self):
        for key, metric in self.metrics.items():
            metric.reset_states()

    def _reset_epoch_metrics(self):
        for key, metric in self.epoch_metrics.items():
            metric.reset_states()

    def _update_metrics(self, **kwargs):
        """
        :param kwargs: some parameter we can change for future
        :return: None (Update state of metrics)
        """
        # get parameter
        loss = kwargs['loss']
        self.metrics['loss'].update_state(loss)

    def _update_metrics_epoch(self, **kwargs):
        loss = kwargs['loss']
        self.epoch_metrics['loss'].update_state(loss)

    def _checkpoint(self):

        def _check_value(v1, v2, mode):
            if (v1 < v2) & (mode == 'min'):
                return True
            elif (v1 > v2) & (mode == 'max'):
                return True
            else:
                return False

        # Get previous and current monitor values.
        previous = self.schedule['monitor_value'].numpy()
        current = self.monitor.result()

        if previous == 0.0:
            self.schedule['monitor_value'].assign(current)

        if _check_value(current, previous, self.mode):
            print("\nMonitor value improved from {:.4f} to {:.4f}.".format(previous, current))

            # Update the schedule.
            self.schedule['monitor_value'].assign(current)

            # And save the model.
            best_model_path = self.scout.save()
            print("\nBest model found and saved: {}".format(best_model_path))
        else:
            print("\nMonitor value not improved: {:.4f}, latest: {:.4f}.".format(previous, current))

        ckpt_path = self.manager.save()
        self._reset_metrics()
        print(f"\nCheckpoint saved at global step {self.schedule['step']}, to file: {ckpt_path}")

    def _log_to_tensorboard_epoch(self, **kwargs):
        char_acc = kwargs['char_acc']
        str_acc = kwargs['str_acc']
        epoch = int(self.schedule['epoch'])
        epoch_train_loss = self.epoch_metrics['loss'].result()

        with self.clerk.as_default():
            tf.summary.scalar("epoch_loss/train", epoch_train_loss, step=epoch)
            tf.summary.scalar("char_acc/train", char_acc, step=epoch)
            tf.summary.scalar("str_acc/train", str_acc, step=epoch)

    def _log_to_tensorboard(self):
        current_step = int(self.schedule['step'])
        train_loss = self.metrics['loss'].result()
        lr = self.optimizer._decayed_lr('float32')

        with self.clerk.as_default():
            tf.summary.scalar("loss", train_loss, step=current_step)
            tf.summary.scalar("learning rate", lr, step=current_step)

    def restore(self, weights_only, from_scout):
        """
        Restore training process from previous training checkpoint.
        Args:
            weights_only: only restore the model weights. Default is False.
            from_scout: restore from the checkpoint saved by model scout.
        """
        if from_scout:
            # scout duoc dung de luu best model
            latest_checkpoint = self.scout.latest_checkpoint
        else:
            # manager duoc dung de luu qua trình training
            latest_checkpoint = self.manager.latest_checkpoint

        if latest_checkpoint is not None:
            print(f"Checkpoint found: {latest_checkpoint}")
        else:
            print(f"WARNING: Checkpoint not found. Model will be initialized from  scratch.")

        print("Restoring ...")

        if weights_only:
            print("Only the model weights will be restored.")
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(checkpoint).expect_partial()  # hidden warning
        else:
            self.checkpoint_scout.restore(latest_checkpoint)
            self.checkpoint_manager.restore(latest_checkpoint)

        print("Checkpoint restored: {}".format(latest_checkpoint))

    def export(self, export_dir):
        print("Saving model to {} ...".format(export_dir))
        self.model.save(export_dir)
        print("Model saved at: {}".format(export_dir))

    def _evaluate(self, data):
        batch_images, batch_targets, encoder_masks, look_ahead_masks = data
        encoder_masks = tf.cast(encoder_masks, dtype=tf.float32)
        look_ahead_masks = tf.cast(look_ahead_masks, dtype=tf.float32)
        for i in range(self.max_length_sequence):  # vocab target
            predictions = self.model([batch_images, encoder_masks, batch_targets, look_ahead_masks], training=False)
            loss = self.loss_fn(predictions, batch_targets)
            acc = get_accu(predictions, batch_targets)
        return loss, acc

    def inference(self, image):
        output = np.zeros((1, 1))
        for i in range(self.max_length_sequence):
            enc_padding_mask = None
            combined_mask = create_look_ahead_mask(i + 1)
            dec_padding_mask = None

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.model((np.array([image]), output),
                                                        enc_padding_mask=enc_padding_mask,
                                                        look_ahead_mask=combined_mask,
                                                        dec_padding_mask=dec_padding_mask,
                                                        training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            output = tf.concat([output[:, :-1], predicted_id, output[:, -1:]], axis=-1)
        return output

    def evalute_specifice_dataset(self, eval_train, eval_val, train_steps_per_epoch):
        dict_result = {
            'train': None,
            'validate': None,
        }
        if eval_train:
            logging.info('evaluate on train set')
            batch_total_loss = 0
            cnt_true_char = 0
            sum_batch = 0
            with tqdm(total=train_steps_per_epoch, initial=0, ascii="->", colour='#1cd41c', position=0,
                      leave=True) as pbar:
                for index in range(0, train_steps_per_epoch):
                    data = self.train_dataloader.next_batch()
                    batch_loss, batch_acc = self._evaluate(data)
                    batch_total_loss += batch_loss
                    cnt_true_char += batch_acc
                    sum_batch += data[0].shape[0]

                    pbar.update(1)
                    pbar.set_postfix({
                        "accuracy": "{:.4f}".format(batch_acc)
                    })

            train_total_loss = batch_total_loss / sum_batch
            train_char_acc = cnt_true_char / sum_batch

            dict_result['train'] = {
                'total_loss': train_total_loss,
                'accuracy': train_char_acc
            }

        if eval_val:
            dict_result['validate'] = {
                'total_loss': 0,
                'accuracy': 0
            }

        return dict_result

    def train(self, epochs, steps_per_epoch):
        initial_epoch = self.schedule['epoch'].numpy()  # loading from schedule
        global_step = self.schedule['step'].numpy()
        initial_step = global_step % initial_epoch
        print("Resume training from global step: {}, epoch: {}".format(global_step, initial_epoch))
        print("Current step is: {}".format(initial_step))

        for epoch in range(initial_epoch, epochs):
            print(f"\n*Epoch {epoch}/{epochs}")
            progress_bar = tqdm(total=steps_per_epoch, initial=initial_step,
                                ascii="->", colour='#1cd41c')

            for index in range(initial_step, steps_per_epoch):
                data = self.train_dataloader.next_batch()
                batch_loss, batch_acc = self._train_step(data)

                # update metrics
                self._update_metrics(loss=batch_loss)
                self._update_metrics_epoch(loss=batch_loss)

                # update process
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": "{:.4f}".format(batch_loss.numpy()),
                    "accuracy": "{:.4f}".format(batch_acc.numpy())
                })

                # update schedule
                self.schedule['step'].assign_add(1)

                # Log and checkpoint the model
                if int(self.schedule['step']) % self.save_freq == 0:
                    self._log_to_tensorboard()
                    self._checkpoint()

            # evaluate on train set
            if epoch % 5 == 0:
                print("\nEvaluate Phase.")
                result_eval = self.evalute_specifice_dataset(eval_train=True,
                                                             eval_val=False,
                                                             train_steps_per_epoch=steps_per_epoch)

                print('Information on Train_set:')
                print('Total loss:        {:.6f}'.format(result_eval['train']['total_loss']))
                print('Category accuracy: {:.6f}'.format(result_eval['train']['accuracy']))

                # # logs tensorboard to epoch
                # self._log_to_tensorboard_epoch(char_acc=result_eval['train']['char_acc'],
                #                                str_acc=result_eval['train']['str_acc'])

            # update epoch
            self.schedule['epoch'].assign_add(1)

            # close process bar
            progress_bar.close()
            initial_step = 0
            self._reset_epoch_metrics()
