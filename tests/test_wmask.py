import numpy as np
import tensorflow as tf

from models import WMask
from tests import TestBase
from utils import matrix_mapping


class WMaskTest(TestBase):
    """
        This class tests the functionality of the WMask model.
        This test suite can be ran with:
            python -m unittest -q tests.WMaskTest
        """

    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the WMask model.
        """
        super(WMaskTest, self).setUp()
        mh_index_map, unk_matrix_id = matrix_mapping.create_matrix_mapping(train_mh=self._db.mh_set,
                                                                           unk_vec_id=self._db.unk_vector_id)
        self._comp_model = WMask(embedding_size=2,
                      mh_index_map=mh_index_map,
                      unk_matrix_id=unk_matrix_id,
                      nonlinearity=tf.identity,
                      dropout_rate=0.0)

    def test_composition(self):
        """
        Tests if the composition method itself is correct. The composition method is
        p  = W[t_u * u; t_v * v] + b

        """
        u = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        v = np.array([[2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        t_v = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        t_u = np.array([[2.0, 2.0], [3.0, 2.0], [1.0, 3.0]])
        W = np.array([[3.0, 3.0], [2.0, 3.0], [2.0, 4.0], [3.0, 3.0]])
        b = np.array([3.0, 2.0])
        result = np.array([[20.0, 25.0], [49.0, 60.0], [32.0, 39.0]])
        with tf.Session() as sess:
            prediction = sess.run(self._comp_model.compose(u=u, v=v, t_u=t_u, t_v=t_v, W=W, b=b))
        np.testing.assert_equal(prediction, result)
