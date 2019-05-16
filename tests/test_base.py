import abc
import unittest
from pathlib import Path

import tensorflow as tf
import numpy as np

import data
from training_graph import TrainingGraph, RunMode


class TestBase(unittest.TestCase, metaclass=abc.ABCMeta):
    """
    This class is the base class all tests for composition models inherit from. Every test requires a setUp() and
    tearDown() method, that are executed before each test and that are needed to construct the data and general
    properties for each model and to clear the tensorflow defaulf graph.
    The data for all tests is read from the 'test_data' directory. For further information about the test data, please
    read the readme in that directory. Furthermore all tests are required to test the composition function.
    Run all test classes with:
         python -m unittest discover -v
    """

    @abc.abstractmethod
    def setUp(self):
        """
        This sets up the data and properties (e.g. embedding dimension, batch size) all models are built on. These properties
        are fixed and the same for all test classes inheriting from this class.
        """
        self._test_data_dir = Path(__file__).resolve().parents[1]
        embeddings_file = str(self._test_data_dir.joinpath("test_data").joinpath("embeddings.txt"))
        train_dataset = str(self._test_data_dir.joinpath("test_data").joinpath("train_data.txt"))
        validation_dataset = str(self._test_data_dir.joinpath("test_data").joinpath("valid_data.txt"))

        self._unknown_word_key = "<unk>"

        self._embedding_model = data.read_word_embeddings(embeddings_file, self._unknown_word_key)
        self._word_index = self._embedding_model.wv.vocab
        self._lookup = self._embedding_model.wv.syn0
        self._embedding_dim = self._lookup.shape[1]
        self._batch_size = 3

        self._db = data.generate_instances(batch_size=self._batch_size, file_path=train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, separator=" ")
        self._vd = data.generate_instances(batch_size=self._batch_size, file_path=validation_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, separator=" ")
        self._comp_model = None

    def tearDown(self):
        """
        This method resets and clears the tensorflow graph after every test.
        """
        tf.reset_default_graph()

    def init_model(self, model):
        """
        Initializes a given composition model.
        :param model: a composition model
        :return: an initialized composition model
        """
        return model.create_architecture()

    def create_training_graph(self, comp_model, learning_rate=0.01, regularization=0.0):
        """
        Constructs a TrainingGraph that is needed to train a composition model and to compute a loss. The learning_rate and
        the amount of regularization can be specified for each case. The composition model needs to be given as an argument.
        :param model: a composition model
        :param learning_rate: learning rate (a float)
        :param regularization: a float that defines the amount of regularization, if set to 0.0 no regularizer is applied
        :return: a computational graph that can be used for training a composition model
        """
        self.init_model(comp_model)
        train_model = TrainingGraph(composition_model=comp_model,
                                    batch_size=None,
                                    learning_rate=learning_rate,
                                    run_mode=RunMode.training,
                                    alpha=regularization)
        return train_model

    def train_model(self, training_model, epochs):
        """
        This method trains a given training graph for a number of epochs. It returns a list of all losses and the final
        result of the last batch of the data.
        :param training_model: a tensorflow graph defining the training model that can be used for training
        :param epochs: an integer defining the number of epochs the model should be trained
        :return: the last batch of predictions for the training data, a list of all losses
        """
        losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_model.model.lookup_init:self._lookup})
            for epoch in range(epochs):
                for batch_idx in range(self._db.no_batches):
                    p, loss = sess.run([training_model.predictions, training_model.loss],
                                       feed_dict={training_model.original_vector: self._db.compound_batches[batch_idx],
                                                  training_model.model._u: self._db.modifier_batches[batch_idx],
                                                  training_model.model._v: self._db.head_batches[batch_idx]})
                    losses.append(loss)
        return p, losses

    def test_architecture(self):
        """
        Tests if the model can be constructed, feed in batches of data, and get the desired result. The shape of the batch of
        the composed representation should have the same shape as the batch of the input representation.
        """
        if self._comp_model.is_trainable:
            training_graph = self.create_training_graph(self._comp_model)
            result, loss = self.train_model(training_graph, epochs=1)
            np.testing.assert_equal(result.shape, [self._batch_size, self._embedding_dim])

    def test_normalized_architecture(self):
        """
        Tests if the model returns a normalized output (magnitude is 1).
        """
        if self._comp_model.is_trainable:
            training_graph = self.create_training_graph(self._comp_model)
            result, loss = self.train_model(training_graph, epochs=1)
            magnitude = np.linalg.norm(result[0])
            np.testing.assert_almost_equal(magnitude, 1)

    def test_lookup_invalid_data(self):
        """
        Tests if the lookup function does raise an exception for word indices that are not in the vocabulary.
        As the vocabulary in ./test_data/embeddings.txt only contains 18 words, the word index 34 is not valid and
        should raise and exception.
        """
        invalid_batch = np.array([34, 34, 34])
        with tf.Session() as sess:
            self.init_model(self._comp_model)
            sess.run(tf.global_variables_initializer(), feed_dict={self._comp_model.lookup_init: self._lookup})
            self.assertRaises(Exception, lambda: sess.run([self._comp_model.architecture],
                                                          feed_dict={
                                                              self._comp_model._u: self._db.compound_batches[0],
                                                              self._comp_model._v: invalid_batch
                                                          }))
    def test_lookup_unknown_word(self):
        """
        Tests if an unknown word is looked up correctly and mapped to the unknown vector. The validation data in
        ./test_data contains an unknown word (Zitrone), that needs to be mapped to the unknown vector ([0.0, 1.0])
        """
        with tf.Session() as sess:
            self.init_model(self._comp_model)
            sess.run(tf.global_variables_initializer(), feed_dict={self._comp_model.lookup_init: self._lookup})
            u = sess.run([self._comp_model.embeddings_u],
                     feed_dict={
                         self._comp_model._u: self._vd.modifier_batches[0],
                         self._comp_model._v: self._vd.head_batches[0]})
            np.testing.assert_equal([0.0, 1.0], u[0][0])

    @abc.abstractmethod
    def test_composition(self):
        """
        Each model needs to test the result of the model-specific composition function.
        """
        #return

    @property
    def comp_model(self):
        """
        This property stores the composition model instantiated by each subclass
        """
        return self._comp_model
