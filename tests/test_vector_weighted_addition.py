import numpy as np
import tensorflow as tf

from models import VectorWeightedAddition
from tests import TestBase


class VectorWeightedAdditionTest(TestBase):
    """
    This class tests the functionality of the VectorWeightedAddition model.
    This test suite can be ran with:
        python -m unittest -q tests.VectorWeightedAdditionTest
    """

    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the VectorWeightedAddition model.
        """
        super(VectorWeightedAdditionTest, self).setUp()
        self._comp_model = VectorWeightedAddition(embedding_size=2)

    def test_composition(self):
        """
        Tests the composition function of the VectorWeightedAddition model
        The function is
        p = a * u + b * v
        """
        u = tf.convert_to_tensor(np.array([2, 3]), np.int64)
        v = tf.convert_to_tensor(np.array([3, 4]), np.int64)
        a = tf.convert_to_tensor(np.array([3, 1]), np.int64)
        b = tf.convert_to_tensor(np.array([2, 3]), np.int64)
        result = np.array([12, 15])
        with tf.Session() as sess:
            p = sess.run(self._comp_model.compose(u, v, a, b))
        np.testing.assert_allclose(result, p, True)

