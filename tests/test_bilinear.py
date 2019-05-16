import numpy as np
import tensorflow as tf

from tests import TestBase
from models import BiLinear


class BiLinearTest(TestBase):
    """
    This class tests the functionality of the Bilinear model.
    This test suite can be ran with:
        python -m unittest -q tests.BiLinearTest
    """
    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the Bilinear model.
        """
        super(BiLinearTest, self).setUp()
        self._comp_model = BiLinear(embedding_size=2, dropout_bilinear_forms=0.0, dropout_matrix=0.0, nonlinearity=tf.identity)

    def test_composition(self):
        """Test if the composition function itself gives the correct output."""
        u = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        v = np.array([[2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        bilinear_forms = np.full(shape=(2, 2, 2), fill_value=2.0)
        W = np.full(shape=(4, 2), fill_value=2.0)
        W[1][0] = 1
        W[2][0] = 3
        W[3][0] = 5
        b = np.array([3.0, 3.0])
        result = [[29.0, 25.0], [57.0, 51.0], [38.0, 33.0]]
        with tf.Session() as sess:
            comp = sess.run(self._comp_model.compose(u, v, bilinear_forms, W, b))
        np.testing.assert_allclose(comp, result)

