import unittest
from pathlib import Path

import tensorflow as tf
import numpy as np

from training_graph import TrainingGraph, RunMode
from models import ScalarWeightedAddition, VectorWeightedAddition, Matrix, WMask, FullLex, TransWeight
from models.fulllex import Regularizer
import data
from utils import matrix_mapping


class TrainingGraphTest(unittest.TestCase):
    """
    This test suite can be ran with:
            python -m unittest -q tests.TrainingGraphTest
    """

    def setUp(self):
        """
        Constructs the data and reads in the embeddings from the example data in the test_data directory
        """
        test_data_dir = Path(__file__).resolve().parents[1]
        embeddings_file = str(test_data_dir.joinpath("test_data").joinpath("embeddings.txt"))
        train_dataset = str(test_data_dir.joinpath("test_data").joinpath("train_data.txt"))

        self._unknown_word_key = "<unk>"
        word_embeddings = data.read_word_embeddings(embeddings_file, self._unknown_word_key)
        self._word_index = word_embeddings.wv.vocab

        self._db = data.generate_instances(batch_size=3, file_path=train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, separator=" ")
        self._mh_index_map, self._unk_matrix_id = matrix_mapping.create_matrix_mapping(train_mh=self._db.mh_set,
                                                                           unk_vec_id=self._db.unk_vector_id)
        self._lookup = word_embeddings.wv.syn0
        tf.set_random_seed(1)

    def tearDown(self):
        tf.reset_default_graph()

    def get_train_model(self, model, alpha):
        """
        Creates a training graph for a composition model
        :param model: a composition model
        :param alpha: the amount of regularization
        :return: a Tensorflow Graph that defines the training graph for a given composition model
        """
        model.create_architecture()
        train_model = TrainingGraph(composition_model=model,
                      batch_size=None,
                      learning_rate=0.01,
                      run_mode=RunMode.training,
                      alpha=alpha)
        return train_model

    def run_model(self, train_model, sess):
        """
        Trains a model for 10 epochs and returns the loss for each epoch
        :param train_model:
        :param sess: a TensorFlow session
        :return: a list of the losses for each epoch
        """
        losses = []
        sess.run(tf.global_variables_initializer(), feed_dict={train_model.model.lookup_init:self._lookup})
        for epoch in range(10):
            train_loss = 0.0
            for tidx in range(self._db.no_batches):
                loss, _ = sess.run(
                    [train_model.loss, train_model.train_op],
                    feed_dict={train_model.original_vector: self._db.compound_batches[tidx],
                               train_model.model._u: self._db.modifier_batches[tidx],
                               train_model.model._v: self._db.head_batches[tidx]})
                train_loss += loss
            train_loss /= self._db.no_batches
            losses.append(train_loss)
        return losses

    def test_loss_scalar(self):
        """Test if the loss decreases for the ScalarWeighted Addition model"""
        with tf.Session() as sess:
            composition_model = ScalarWeightedAddition(embedding_size=2)
            train_model = self.get_train_model(composition_model, alpha=0.0)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)

    def test_loss_vector(self):
        """Test if the loss decreases for the VectorWeightedAddition model"""
        with tf.Session() as sess:
            composition_model = VectorWeightedAddition(embedding_size=2)
            train_model = self.get_train_model(composition_model, alpha=0.0)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)

    def test_loss_matrix(self):
        """Test if the loss decreases for the Matrix model"""
        with tf.Session() as sess:
            composition_model = Matrix(embedding_size=2,nonlinearity=tf.identity,dropout_rate=0.0)
            train_model = self.get_train_model(composition_model, alpha=0.0)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)

    def test_loss_wmask(self):
        """Test if the loss decreases for the WMask model"""
        with tf.Session() as sess:
            composition_model = WMask(embedding_size=2,
                                     mh_index_map=self._mh_index_map,
                                     unk_matrix_id=self._unk_matrix_id,
                                     nonlinearity=tf.identity,
                                     dropout_rate=0.0)
            train_model = self.get_train_model(composition_model, alpha=0.0)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)

    def test_loss_fulllex(self):
        """Test if the loss decreases for the FullLex model"""
        with tf.Session() as sess:
            composition_model = FullLex(embedding_size=2,
                                        mh_index_map=self._mh_index_map,
                                        unk_matrix_id=self._unk_matrix_id,
                                        nonlinearity=tf.identity,
                                        dropout_rate=0.0,
                                        regularizer=Regularizer.l1_regularizer)
            train_model = self.get_train_model(composition_model, alpha=0.0)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)

    def test_loss_transweight(self):
        """Test if the loss decreases for the TransWeight model"""
        with tf.Session() as sess:
            composition_model = TransWeight(embedding_size=2, nonlinearity=tf.identity, dropout_rate=0.0, transforms=2)
            train_model = self.get_train_model(composition_model, alpha=0.0)
            losses = self.run_model(train_model, sess)
        np.testing.assert_equal(losses[0] > losses[9], True)


    def testDefaultRegularizer(self):
        """test if a normal composition model has a default regularizer of 0.0 which
         leads to no change between normal loss and regularized loss"""

        matrix_model = Matrix(embedding_size=2,nonlinearity=tf.identity,dropout_rate=0.0)
        training_graph = self.get_train_model(model=matrix_model, alpha=0.5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_graph.model.lookup_init:self._lookup})
            for epoch in range(3):
                for tidx in range(self._db.no_batches):
                    loss, reg_loss, _ = sess.run(
                        [training_graph.loss, training_graph.reg_loss, training_graph.train_op],
                        feed_dict={training_graph.original_vector: self._db.compound_batches[tidx],
                                   training_graph.model._u: self._db.modifier_batches[tidx],
                                   training_graph.model._v: self._db.head_batches[tidx]})

                    np.testing.assert_equal(loss, reg_loss)

