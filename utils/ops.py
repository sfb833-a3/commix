import tensorflow as tf
import numpy as np


def bias_variable(val, shape):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)

def uv_affine_transform(u, v, W, b):
    """
    Concatenate u and v, and apply an affine transformation.

    W[u;v] + b

    u and v should have shape [batch_size, embed_size].
    """
    concatenation = tf.concat(values=[u, v], axis=1)
    return tf.nn.xw_plus_b(concatenation, W, b)


def identity_initialisation_matrix(table_size, embedding_dim):
    """initialize to a matrix of eye matrices of embedding size, add random noise from normal distribution"""
    with tf.device("/cpu:0"):
        gauss_matrix = tf.random_normal(shape=[table_size, embedding_dim, embedding_dim],
                                        mean=0, stddev=0.0001)
        eye_matrix = tf.eye(embedding_dim, batch_shape=[table_size])
        init_matrix = tf.add(eye_matrix, gauss_matrix)
    return init_matrix


def identity_initialisation_vector(table_size, embedding_dim):
    """initialize to a matrix of vectors filled with 1 of embedding size, add random noise from normal distribution"""
    with tf.device("/cpu:0"):
        init_matrix = tf.add(tf.fill(dims=[table_size, embedding_dim], value=1.0),
                             tf.random_normal(shape=[table_size, embedding_dim],
                                              mean=0, stddev=0.0001))
    return init_matrix

def init_index_hash(unk_matrix_id, name):
    """:returns a MutableDenseHashTable that maps indices of vector representations to indices from a
    fixed size tensor containing only the modifiers and heads in the train data
    during lookup, if a particular input vector index (key) is missing from the hashtable, 
    it is mapped to the default value, unk_matrix_id. Because this id collides with the default value for empty_key
    (default of TensorFlow is 0, the value is changed to -10)
    hash table can be updated after training with the validation / test data indices that will be mapped to the matrix
    indices of their nearest neighbours.

    """
    index_hash = tf.contrib.lookup.MutableDenseHashTable(key_dtype=tf.int64, value_dtype=tf.int64,deleted_key=-1,
                                                         default_value=unk_matrix_id, empty_key=-10, name=name)
    return index_hash

def init_hashtable(mh_map, index_hash, sess):
   """
   initializes / updates the hash table that is used for looking up matrix inidces for vector indices. This can be used when
   the mapping should be changed after training the model.
   :param mh_map: the new mapping that should be used in the mode
   :param index_hash: the current mapping of the composition model that should be updated
   :param sess: the current tensorflow session
   """
   mh_map_keys = tf.constant(np.array(list(mh_map.keys())), tf.int64)
   mh_map_vals = tf.constant(np.array(list(mh_map.values())), tf.int64)
   new_index_hash = index_hash.insert(mh_map_keys, mh_map_vals)
   sess.run(new_index_hash)
