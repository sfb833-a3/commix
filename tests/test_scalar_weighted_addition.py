import numpy as np
import tensorflow as tf
from models import ScalarWeightedAddition
from tests import TestBase
import unittest


class ScalarWeightedAdditionTest(TestBase):
    """
    This class tests the functionality of the ScalarWeightedAddition model.
    This test suite can be ran with:
        python -m unittest -q tests.ScalarWeightedAdditionTest
    """
    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the ScalarWeightedAddition model.
        """
        super(ScalarWeightedAdditionTest, self).setUp()
        model = ScalarWeightedAddition(embedding_size=2)
        self._comp_model = model


    def test_composition(self):
        """
        Tests if the composition method itself is correct. The composition method is
        p = a*u + b*v
        a and b are scalars. In this example it is
        (2,3) * 2 + (3,4)*2 = (4,6) + (6,8) = (10,14)
        """
        u = tf.convert_to_tensor(np.array([2, 3]), dtype=tf.int64)
        v = tf.convert_to_tensor(np.array([3, 4]), dtype=tf.int64)
        alpha = tf.constant(2, dtype=tf.int64)
        beta = tf.constant(2, dtype=tf.int64)
        result = np.array([10, 14])
        model = ScalarWeightedAddition(embedding_size=2)
        with tf.Session() as sess:
            p = sess.run(model.compose(u, v, alpha, beta))
        np.testing.assert_allclose(result, p, True)


if __name__ == "__main__":
    unittest.main()
