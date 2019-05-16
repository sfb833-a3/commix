import tensorflow as tf
import numpy as np

from models import TransWeightMatrix
from tests import TestBase

class TransWeightMatrixTest(TestBase):
    """
    This class tests the functionality of the TransWeightMatrix model.
    This test suite can be ran with:
        python -m unittest -q tests.TransWeightMatrixTest
    """

    def setUp(self):
        super(TransWeightMatrixTest, self).setUp()
        self._comp_model = TransWeightMatrix(embedding_size=2, nonlinearity=tf.identity, dropout_rate=0.0, transforms=2)

    def test_weight(self):
        """
        Test that the weighting is correctly performed
        If [[A B C] are two transformed representations and [[g h i] is the transformations weighting matrix,
            [D E F]]                                          [j k l]]  where each row can be considered as one weight
                                                                        vector for the corresponding tranformation,

        then the elements of the composed representation [p_0 p_1 p_2] are obtained as:
        p_0 = A*g + D*j
        p_1 = B*h + E*k
        p_2 = C*i + F*l
        """

        t = np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32')

        W = np.array([[[2, 3, 4], [1, 2, 3]]], dtype='float32')

        b = np.full(shape=(3,), fill_value=0.0, dtype='float32')

        expected_p = np.array([[4, 3, 3]])

        with tf.Session() as sess:
            p = sess.run(
                self.comp_model.weight(
                    reg_uv=t,
                    W=W,
                    b=b))

        np.testing.assert_allclose(p, expected_p)

    def test_composition(self):
        pass
