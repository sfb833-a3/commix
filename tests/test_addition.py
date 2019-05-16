import numpy as np
import tensorflow as tf

from tests import TestBase
from models import Addition
from training_graph import TrainingGraph, RunMode

class AdditionTest(TestBase):
    """
    This class tests the functionality of the Addition model.
    This test suite can be ran with:
        python -m unittest -q tests.AdditionTest
    """

    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the ScalarWeightedAddition model.
        """
        super(AdditionTest, self).setUp()
        model = Addition(embedding_size=2)
        self._comp_model = model

    def test_composition(self):
        """
        Test if the composition function returns correct result
        p = u + v
        """
        u = np.array([2,3])
        v = np.array([3,4])
        result = np.array([5,7])
        with tf.Session() as sess:
            p = sess.run(self._comp_model.compose(u, v))
        np.testing.assert_allclose(result, p, True)

    def test_trainable_property(self):
        """Test if the is_trainable property of the AdditionModel is False"""
        np.testing.assert_equal(self._comp_model.is_trainable, False)

    def test_architecture(self):
        """"Test if the whole architecture of the addition model can be ran
         and produces a result of the correct shape"""
        self._comp_model.create_architecture()
        training_model = TrainingGraph(composition_model=self._comp_model,
                      batch_size=None,
                      learning_rate=0.0,
                      run_mode=RunMode.validation,
                      alpha=0.0)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_model.model.lookup_init: self._lookup})
            p, loss = sess.run([training_model.model.architecture, training_model.loss],
                               feed_dict={training_model.original_vector: self._db.compound_batches[0],
                                          training_model.model._u: self._db.modifier_batches[0],
                                          training_model.model._v: self._db.head_batches[0]})
        np.testing.assert_equal(p.shape, [self._batch_size, self._embedding_dim])

    def test_architecture_normalized(self):
        """
        Tests if the model returns a normalized output (magnitude is 1).
        """
        self._comp_model.create_architecture()
        training_model = TrainingGraph(composition_model=self._comp_model,
                                       batch_size=None,
                                       learning_rate=0.0,
                                       run_mode=RunMode.validation,
                                       alpha=0.0)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_model.model.lookup_init: self._lookup})
            p, p_norm = sess.run([training_model.model.architecture, training_model.model.architecture_normalized],
                               feed_dict={training_model.original_vector: self._db.compound_batches[0],
                                          training_model.model._u: self._db.modifier_batches[0],
                                          training_model.model._v: self._db.head_batches[0]})
        for composed_rep in p_norm:
            magnitude = np.linalg.norm(composed_rep)
            np.testing.assert_almost_equal(magnitude, 1)
