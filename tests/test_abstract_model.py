import unittest

import numpy as np
import tensorflow as tf

from models import AbstractModel
from models import ScalarWeightedAddition


class ConcreteModel2(AbstractModel):
    """This is a test class which defines an incomplete concrete model that inherits from the abstract model."""
    def __init__(self):
        pass

    def create_architecture(self):
        return self._architecture


class AbstractModelTest(unittest.TestCase):
    """
    This class tests the functionality of the abstract model.
    This test suite can be ran with:
        python -m unittest -q tests.AbstractModelTest
    """

    # Initialize a concrete Model
    def init_model(self):
        return ConcreteModel2()

    def tearDown(self):
        tf.reset_default_graph()

    def test_exception(self):
        """
        Trying to initialize the incomplete concrete model should throw a Type Exception
        because the concrete implementation lacks some properties and methods
        that must be implemented when inheritance from the AbstractModel is wanted.
        """
        self.assertRaises(TypeError, lambda: self.init_model())

    def test_normalization(self):
        """Test if normalization method from the AbstractModel results in a batch of vectors of magnitude 1"""
        u = np.array([np.array([0.7, 1.2]), np.array([0.5, 1.6])])
        with tf.Session() as sess:
            n = sess.run(AbstractModel.l2_normalization_layer(u, axis=1))
        magnitude = np.linalg.norm(n, axis=1)
        np.testing.assert_allclose(magnitude, np.array([1.0, 1.0]))

    def test_trainable_property(self):
        """Test if the default property a concrete model was inherited correctly form the abstract model"""
        scalar_weighted_addition_model = ScalarWeightedAddition(10)
        np.testing.assert_equal(scalar_weighted_addition_model.is_trainable, True)

    def test_feed_in_input_vectors(self):
        """Test if the input vectors can be fed into any model that inherited from the abstract model. Test if the lookup
        function is also performed because the model inherited from the AbstractModel. Test that the composition can be
        constructed with the model and that all properties (vectors, embeddings, lookup table) have a concrete and correct shape.
        """
        lookup = np.full(shape=[10, 2], fill_value=0.5)
        u = np.array([6, 5])
        v = np.array([8, 1])
        model = ScalarWeightedAddition(2)
        model.create_architecture()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={model.lookup_init:lookup})
            model_lookup, emb_u, emb_v, architecture, architecture_normalized = sess.run([model.lookup, model.embeddings_u,
                                                                            model.embeddings_v, model.architecture,
                                                                            model.architecture_normalized],
                                                             feed_dict={
                                                                 model.lookup: lookup,
                                                                 model._u: u,
                                                                 model._v: v})
        np.testing.assert_equal(lookup.shape, model_lookup.shape)
        np.testing.assert_equal(emb_u.shape, [2, 2])
        np.testing.assert_equal(emb_v.shape, [2, 2])
        np.testing.assert_equal(architecture.shape, [2, 2])

