import tensorflow as tf
import numpy as np

from models import TransWeightTransformations
from tests import TestBase

class TransWeightTransformationsTest(TestBase):
    """
    This class tests the functionality of the TransWeightTransformations model.
    This test suite can be ran with:
        python -m unittest -q tests.TransWeightTransformationsTest
    """

    def setUp(self):
        super(TransWeightTransformationsTest, self).setUp()
        self._comp_model = TransWeightTransformations(embedding_size=2, nonlinearity=tf.identity, dropout_rate=0.0, transforms=2)

    def test_weight(self):
        """
        Test that the weighting is correctly performed
        If [[A B C] are two transformed representations and [j k] is the transformations weighting vector of
            [D E F]]                                        dimension [1 x t]
        than the elements of the composed representation [p_0 p_1 p_2] are obtained as:
        p_0 = A*j + D*k
        p_1 = B*j + E*k
        p_2 = C*j + F*k
        """

        t = np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32')
        W = np.array([[2], [3]], dtype='float32')

        b = np.full(shape=(3,), fill_value=0.0, dtype='float32')

        expected_p = np.array([[4, 2, 3]])

        with tf.Session() as sess:
            p = sess.run(
                self.comp_model.weight(
                    reg_uv=t,
                    W=W,
                    b=b))

        np.testing.assert_allclose(p, expected_p)

    def test_composition(self):
        pass
