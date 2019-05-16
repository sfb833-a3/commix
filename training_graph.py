from enum import Enum

import tensorflow as tf

print(tf.__version__)

class RunMode(Enum):
    training = 1
    validation = 2
    prediction = 3

class TrainingGraph():
    def __init__(self, composition_model, batch_size, learning_rate, run_mode, alpha=0.0):
        self._model = composition_model
        self._is_training = self.model._is_training

        self._original_vector = tf.placeholder(dtype=tf.int64, shape=[batch_size])
        original_embeddings = tf.nn.embedding_lookup(params=self._model.lookup, ids=self._original_vector)

        self._predictions = self._model._architecture_normalized

        self._loss = self._reg_loss = tf.losses.cosine_distance(labels=original_embeddings,
                                               predictions=self._predictions, axis=1, reduction=tf.losses.Reduction.SUM)

        self._train_op = tf.no_op()

        if run_mode is RunMode.training:
            if alpha > 0.0:
                self._reg_loss += alpha*composition_model.regularization()
                self._train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self._reg_loss)
            else:
                self._train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self._loss)


        #summaries for tensorboard
        tf.summary.scalar("learning rate", learning_rate)
        tf.summary.scalar("loss", self._loss)

    @property
    def is_training(self):
        return self._is_training

    @property
    def model(self):
        return self._model

    @property
    def original_vector(self):
        return self._original_vector

    @property
    def architecture(self):
        return self._architecture

    @property
    def loss(self):
        return self._loss

    @property
    def reg_loss(self):
        return self._reg_loss

    @property
    def predictions(self):
        return self._predictions

    @property
    def train_op(self):
        return self._train_op
