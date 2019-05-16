import tensorflow as tf
import numpy as np

from models import TransWeightFeatures
from tests import TestBase


class TransWeightFeaturesTest(TestBase):
    """
    This class tests the functionality of the TransWeightFeatures model.
    This test suite can be ran with:
        python -m unittest -q tests.TransWeightFeaturesTest
    """
    def setUp(self):
        super(TransWeightFeaturesTest, self).setUp()        
        self._comp_model = TransWeightFeatures(embedding_size=2, nonlinearity=tf.identity, dropout_rate=0.0, transforms=2)

    def test_weight(self):
        """
        Test that the weighting is correctly performed
        If [[A B C] are two transformed representations and [g h i] is the transformations weighting vector
            [D E F]]
        than the elements of the composed representation [p_0 p_1 p_2] are obtained as:
        p_0 = A*g + D*g
        p_1 = B*h + E*h
        p_2 = C*i + F*i
        """

        t = np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32')
        W = np.array([[1, 3, 4]])

        b = np.full(shape=(3,), fill_value=0.0, dtype='float32')

        expected_p = np.array([[2, 3, 4]])

        with tf.Session() as sess:
            p = sess.run(
                self.comp_model.weight(
                    reg_uv=t,
                    W=W,
                    b=b))

        np.testing.assert_allclose(p, expected_p)

    def test_composition(self):
        pass
