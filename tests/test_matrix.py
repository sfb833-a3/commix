import numpy as np
import tensorflow as tf

from tests import TestBase
from models import Matrix


class MatrixTest(TestBase):
    """
       This class tests the functionality of the Matrix model.
       This test suite can be ran with:
           python -m unittest -q tests.MatrixTest
       """
    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the Matrix model.
        """
        super(MatrixTest, self).setUp()
        self._comp_model = Matrix(embedding_size=self._embedding_dim,
                                  nonlinearity=tf.identity,
                                  dropout_rate=0.0)


    def test_composition(self):
        """
        Tests if the composition method itself is correct. The composition method is
        p  = W[u;v] + b

        """
        u = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        v = np.array([[2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        W = np.full(shape=(4, 2), fill_value=2.0)
        W[1][0] = 1
        W[2][0] = 3
        W[3][0] = 5
        b = np.array([3.0, 3.0])
        result = np.array([[17, 13], [25, 19], [20, 15]])
        with tf.Session() as sess:
            comp = sess.run(self._comp_model.compose(u, v, W, b, 1))
        np.testing.assert_allclose(comp, result)

