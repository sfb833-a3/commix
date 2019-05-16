import tensorflow as tf
import numpy as np

from models import TransWeight
from tests import TestBase

class TransWeightTest(TestBase):
    """
    This class tests the functionality of the TransWeight model.
    This test suite can be ran with:
        python -m unittest -q tests.TransWeightTest
    """

    def setUp(self):
        super(TransWeightTest, self).setUp()
        self._comp_model = TransWeight(embedding_size=2, nonlinearity=tf.identity, dropout_rate=0.0, transforms=2)

    def test_transformation(self):
        """
        Tests that the t transformations are correctly performed.
        transformations_tensor contains t transformations matrices of size 2nxn, where
        n is the size of the input vectors u and v
        """

        u = np.array([[1, 1, 1]], dtype='float32')
        v = np.array([[1, 0, 0]], dtype='float32')

        transformations_tensor = np.full(shape=(2,6,3), fill_value=0.0, dtype='float32')
        
        two_eyes = np.concatenate((np.eye(3), np.eye(3)), axis=0)
        transformations_tensor[0] = np.copy(two_eyes)
        transformations_tensor[0][2][2] = 0
        transformations_tensor[1] = np.copy(two_eyes)
        transformations_tensor[1][0][0] = 0
        transformations_tensor[1][1][1] = 0
        transformations_tensor[1][3][0] = 0

        transformations_bias = np.full(shape=(2,3), fill_value=0.0, dtype='float32')

        # the two transformation matrices are different, so the resulting
        # transformations are also expected to be different,
        # even if they start off with the same inputs u and v
        expected_t = np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32')
        
        with tf.Session() as sess:        
                t = sess.run(
                    self.comp_model.transform(
                        u=u, 
                        v=v, 
                        transformations_tensor=transformations_tensor, 
                        transformations_bias=transformations_bias))

        np.testing.assert_allclose(t, expected_t)


    def test_weighting(self):
        """
        Test that the weighting is correctly performed
        If [[A B C] are two transformed representations and [[[g h i] are the transformations weighting matrices
            [D E F]]                                          [j k l]
                                                              [m n p]]
                                                             [[q r s]
                                                              [t u v]
                                                              [x y z]]]
        than the elements of the composed representation [p_0 p_1 p_2] are obtained as:
        p_0 = A*g + B*j + C*m + D*q + E*t + F*x
        p_1 = A*h + B*k + C*n + D*r + E*u + F*y
        p_2 = A*i + B*l + C*p + D*s + E*v + F*z
        """

        t = np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32')

        W = np.full(shape=(2,3,3), fill_value=0.0, dtype='float32')
        W[0][0][0] = 1
        W[0][0][1] = 1
        W[0][0][2] = 2

        W[0][1][0] = 2
        W[0][1][1] = 2
        W[0][1][2] = 0

        W[0][2][0] = 0
        W[0][2][1] = 2
        W[0][2][2] = 2

        W[1][0][0] = 2
        W[1][0][1] = 0
        W[1][0][2] = 0

        W[1][1][0] = 0
        W[1][1][1] = 1
        W[1][1][2] = 0

        W[1][2][0] = 1
        W[1][2][1] = 2
        W[1][2][2] = 2


        b = np.full(shape=(3,), fill_value=0.0, dtype='float32')

        expected_p = np.array([[5, 6, 6]])
        
        with tf.Session() as sess:        
                p = sess.run(
                    self.comp_model.weight(
                        reg_uv=t,
                        W=W, 
                        b=b))

        np.testing.assert_allclose(p, expected_p)

    def test_composition(self):
        """
        Testing composition for TW is done by testing the two seps in test_transformations and
        test_weighting.
        """
        pass
