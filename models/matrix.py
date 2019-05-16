import tensorflow as tf

from models import AbstractModel
from utils.ops import uv_affine_transform


class Matrix(AbstractModel):
    """
    Contains the architecture of the Matrix composition model.
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size, nonlinearity, dropout_rate):
        super(Matrix, self).__init__(embedding_size)
        self._nonlinearity = nonlinearity
        self._dropout_rate = dropout_rate

    def create_architecture(self):

        # weightmatrix that should be optimized. shape: 2n x n
        W = tf.get_variable("W", shape=[self.embedding_size * 2, self.embedding_size])
        self._W = tf.layers.dropout(W, rate=self.dropout_rate,
                                    training=self.is_training)

        # biasvector. shape: n
        self._b = tf.get_variable("b", shape=[self.embedding_size])

        self._architecture = self.compose(u=self.embeddings_u, v=self.embeddings_v,
                                          W=self._W, b=self._b, axis=1)

        # adds nonlinearity to the composition
        self._architecture = self.nonlinearity(self._architecture)

        self._architecture_normalized = super(
            Matrix,
            self).l2_normalization_layer(
            self._architecture,
            1)

    def compose(self, u, v, W, b, axis):
        """
        composition of the form:
        p = g(W[v; u] + b)
        takes as input two tensors of any shape (tensors need to have at least rank 2)
        concatenates vector v and u and multiplies the concatenation by weightmatrix.
        adds biasvector b
        """
        return uv_affine_transform(u, v, W, b)

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @property
    def nonlinearity(self):
        return self._nonlinearity
