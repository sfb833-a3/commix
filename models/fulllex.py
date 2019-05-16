from enum import Enum

import tensorflow as tf

from models import AbstractModel
from utils.ops import uv_affine_transform, \
                identity_initialisation_matrix, \
                init_index_hash


class Regularizer(Enum):
    l1_regularizer = 1
    dot_regularizer = 2

class FullLex(AbstractModel):

    """
    Contains the architecture of the FullLexModel
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size, mh_index_map, unk_matrix_id, nonlinearity, dropout_rate, regularizer):
        super(FullLex, self).__init__(embedding_size)
        self._vocab_size = len(mh_index_map)
        self._mh_index_map = mh_index_map
        self._unk_matrix_id = unk_matrix_id
        self._nonlinearity = nonlinearity
        self._dropout_rate = dropout_rate
        self._regularizer = regularizer
        # lookup the wordspecific word matrices
        self._index_hash = init_index_hash(self._unk_matrix_id, "index")

    def create_architecture(self):

        # weightmatrix that should be optimized. shape: 2n x n
        self._W = tf.get_variable("W", shape=[self.embedding_size*2, self.embedding_size])
        # biasvector. shape: n
        self._b = tf.get_variable("b", shape=[self.embedding_size])


        # create a lookuptable that contains the word matrices for all heads / modifier
        self._matrix_lookup = tf.get_variable("matrix_lookup",
                                              initializer=identity_initialisation_matrix(self._vocab_size,
                                                                                         self.embedding_size))

        # maps the vector indices to the matrix indices


        self._matrix_U = tf.nn.embedding_lookup(self._matrix_lookup, self._index_hash.lookup(self._u))
        self._matrix_V = tf.nn.embedding_lookup(self._matrix_lookup, self._index_hash.lookup(self._v))

        self._matrix_U = tf.layers.dropout(self._matrix_U, rate = self.dropout_rate,
                training = self.is_training)
        self._matrix_V = tf.layers.dropout(self._matrix_V, rate = self.dropout_rate,
                training = self.is_training)

        # get the composition
        self._architecture = self.compose(self._embeddings_u, self._embeddings_v, self._matrix_U, self._matrix_V,
                                          self._W, self._b)
        # add nonlinearity to the composition
        self._architecture = self.nonlinearity(self._architecture)

        # l2 normalize the composition
        self._architecture_normalized = super(
            FullLex, self).l2_normalization_layer(
                tensor=self._architecture, axis=1)

    def compose(self, u, v, matrix_U, matrix_V, W, b):
        """
        composition of the form:
        p = p = g(W[Vu;Uv])
        takes as input two tensors of any shape (tensors need to have at least rank 2)
        multiplies each vector by the word matrix of the other word vector.
        concatenates the outcoming vectors multiplies the concatenation by weightmatrix.
        adds biasvector b
        """
        u = tf.expand_dims(u, axis=2)
        v = tf.expand_dims(v, axis=2)
        Uv = tf.matmul(matrix_U, v)
        Vu = tf.matmul(matrix_V, u)
        self._Uv = tf.squeeze(Uv, axis=2)
        self._Vu = tf.squeeze(Vu, axis=2)

        return uv_affine_transform(self._Vu, self._Uv, W, b)

    def l1_regularizer(self, matrix_U, matrix_V):
        """
        implements a regularization method of the form:
        sum(|Av - Iv|) are the current word matrices for that batch and Iv are eye matrices
        calculate how high the total difference between the current matrices and eye matrices is
        :param matrix_U: the current matrix of input 1 of word matrices
        :param matrix_V: the current matrix of input 2 word matrices
        :return: a scalar that defines the total difference
        """
        word_matrices = tf.concat(values=[matrix_U, matrix_V], axis=0)
        I = tf.eye(tf.shape(matrix_V)[1], batch_shape=[tf.shape(word_matrices)[0]])
        sub = tf.subtract(word_matrices, I)
        regularized_sum = tf.reduce_sum(tf.norm(sub, axis=[-2, -1], ord=1), axis=0)
        return regularized_sum

    def dot_regularizer(self, Vu, u, Uv, v):
        """
        Calculates the cosine distance between Vu and u and Uv and v. Returns the sum of those
        distances. So if V and U are identity matrices the distances will be 0 and there will be a penalty of 0.
        :param Vu: matrix product of input 1 and word matrices of input 2
        :param u: matrix of input 1
        :param Uv: matrix product of input 2 and word matrices of input 1
        :param v: matrix of input 2
        :return: a scalar that defines the sum of the cosine distances
        """
        Uv_normalized = super(FullLex, self).l2_normalization_layer(
            tensor=Uv, axis=1)
        Vu_normalized = super(FullLex, self).l2_normalization_layer(
            tensor=Vu, axis=1)
        regularized_sum = tf.add(tf.losses.cosine_distance(Uv_normalized, v, axis=1, reduction=tf.losses.Reduction.SUM),
                                 tf.losses.cosine_distance(Vu_normalized, u, axis=1, reduction=tf.losses.Reduction.SUM))
        return regularized_sum

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def matrix_lookup(self):
        return self._matrix_lookup

    @property
    def matrix_U(self):
        return self._matrix_U

    @property
    def matrix_V(self):
        return self._matrix_V

    @property
    def Uv(self):
        return self._Uv

    @property
    def Vu(self):
        return self._Vu

    @property
    def nonlinearity(self):
        return self._nonlinearity

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def index_hash(self):
        return self._index_hash

    @property
    def regularizer(self):
        return self._regularizer

    def regularization(self):
        if self.regularizer is Regularizer.l1_regularizer:
            return self.l1_regularizer(matrix_U=self._matrix_U, matrix_V=self.matrix_V)
        else:
            return self.dot_regularizer(u=self._embeddings_u, Uv=self._Uv, v=self._embeddings_v, Vu=self._Vu)

