import unittest
import numpy as np
import tensorflow as tf

from utils.ops import uv_affine_transform, \
                identity_initialisation_vector, \
                identity_initialisation_matrix, \
                init_index_hash

class UVAffineTransformationTest(unittest.TestCase):
    """
    This test suite can be ran with:
        python -m unittest -q tests.UVAffineTransformationTest
    """
    def setUp(self):
        tf.enable_eager_execution()

    def test_transformation(self):
        """Test if the affine transformation function returns the correct result"""
        u = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
        v = tf.constant([2, 2, 2, 1, 1, 1], shape=[2, 3])
        W = tf.constant(2, shape=[6, 3])
        b = tf.constant(1, shape=[3])
        t = uv_affine_transform(u, v, W, b)
        result = [[25, 25, 25], [37, 37, 37]]
        np.testing.assert_allclose(t, result)


class InitializationOpsTest(unittest.TestCase):
    """
    This test suite can be ran with:
        python -m unittest -q tests.InitializationOpsTest
    """
    def setUp(self):
        self._table_size = 10
        self._embedding_dim = 100

    def test_matrix_initializer(self):
        """Test if initializer returns a matrix of matrices with correct shape"""
        matrix_table = identity_initialisation_matrix(table_size=self._table_size, embedding_dim=self._embedding_dim)
        np.testing.assert_equal([self._table_size, self._embedding_dim, self._embedding_dim], matrix_table.shape)

    def test_vector_initializer(self):
        """Test if initializer returns a matrix of vectors with correct shape"""
        vector_table = identity_initialisation_vector(table_size=self._table_size, embedding_dim=self._embedding_dim)
        np.testing.assert_equal([self._table_size, self._embedding_dim], vector_table.shape)


class HashTableInitTest(unittest.TestCase):

    """
     This test suite can be ran with:
        python -m unittest -q tests.HashTableInitTest
    only a single test because of Eager Mode issue with initializing the HashTable;
        https://github.com/tensorflow/tensorflow/issues/19626
    """
    
    def setUp(self):
        tf.enable_eager_execution()
        self._index_dict = {}
        self._index_dict[124] = 0
        self._index_dict[345] = 1
        self._index_dict[115] = 2
        self._index_dict[105] = 3 # 105 would be the index of the unk vector

        self._unk_matrix_idx = 3
        self._index_hash = init_index_hash(self._unk_matrix_idx, "index")
        mh_map_keys = tf.constant(np.array(list(self._index_dict.keys())), tf.int64)
        mh_map_vals = tf.constant(np.array(list(self._index_dict.values())), tf.int64)
        self._index_hash.insert(mh_map_keys, mh_map_vals)


    def test_indices(self):
        """Test if the known/unknown indices are mapped as expected: known indices to their matrix idx,
        unknown indices to the unknown matrix idx

        """
        matrix_indices = self._index_hash.lookup(tf.constant([105, 345, 124], shape=[3,], dtype='int64'))
        np.testing.assert_allclose(matrix_indices, tf.constant([3, 1, 0], shape=[3,], dtype='int64'))


if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()
