import tensorflow as tf

from models import AbstractModel
from utils.ops import uv_affine_transform, \
                identity_initialisation_vector, \
                init_index_hash


class WMask(AbstractModel):

    def __init__(self, embedding_size, mh_index_map, unk_matrix_id, nonlinearity, dropout_rate):
        super(WMask, self).__init__(embedding_size)
        self._vocab_size = len(mh_index_map)
        self._mh_index_map = mh_index_map
        self._unk_matrix_id = unk_matrix_id
        self._nonlinearity = nonlinearity
        self._dropout_rate = dropout_rate
        # maps the vector indices to the matrix indices
        self._index_hash = init_index_hash(self._unk_matrix_id, "index")


    def create_architecture(self):

        # weightmatrix that should be optimized. shape: 2n x n
        self._W = tf.get_variable("W", shape=[self.embedding_size * 2, self.embedding_size])
        # biasvector. shape: n
        self._b = tf.get_variable("b", shape=[self.embedding_size])
        # lookuptable(s)

        self._vector_lookup_u = tf.get_variable("vector_lookup_u",
                                                initializer=identity_initialisation_vector(self._vocab_size,
                                                                                           self.embedding_size))
        self._vector_lookup_v = tf.get_variable("vector_lookup_v",
                                                    initializer=identity_initialisation_vector(self._vocab_size,
                                                                                               self.embedding_size))

        # lookup the wordspecific word matrices
        self._t_u = tf.nn.embedding_lookup(self._vector_lookup_u, self._index_hash.lookup(self._u))
        self._t_v = tf.nn.embedding_lookup(self._vector_lookup_v, self._index_hash.lookup(self._v))

        self._t_u = tf.layers.dropout(self._t_u, rate = self.dropout_rate,
                training = self.is_training)
        self._t_v = tf.layers.dropout(self._t_v, rate = self.dropout_rate,
                training = self.is_training)

        # get the composition
        self._architecture = self.compose(u=self._embeddings_u, v=self._embeddings_v, t_u=self._t_u,
                                          t_v=self._t_v, W=self._W, b=self._b)

        # add nonlinearity to the composition
        self._architecture = self.nonlinearity(self._architecture)

        # l2 normalize the composition
        self._architecture_normalized = super(
            WMask, self).l2_normalization_layer(
            tensor=self._architecture, axis=1)

    # composition of the form W[t_u * u; t_v * v] + b)
    def compose(self, u, v, t_u, t_v, W, b):
        u_t_u = tf.multiply(u, t_u)
        v_t_v = tf.multiply(v, t_v)

        return uv_affine_transform(u_t_u, v_t_v, W, b)

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def vector_lookup_u(self):
        return self._vector_lookup_u

    @property
    def vector_lookup_v(self):
        return self._vector_lookup_v

    @property
    def t_u(self):
        return self.t_u

    @property
    def t_v(self):
        return self.t_v

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
    def nonlinearity(self):
        return self._nonlinearity

    @property
    def index_hash(self):
        return self._index_hash
