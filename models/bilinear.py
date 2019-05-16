import tensorflow as tf

from models import AbstractModel
from utils.ops import uv_affine_transform


class BiLinear(AbstractModel):
    """
    Contains the architecture of the BiLinear model.
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """
    def __init__(self, embedding_size, nonlinearity, dropout_bilinear_forms, dropout_matrix):
        super(BiLinear, self).__init__(embedding_size)
        self._nonlinearity = nonlinearity
        self._dropout_bilinear_forms = dropout_bilinear_forms
        self._dropout_matrix = dropout_matrix

    def create_architecture(self):

        # embedding weight matrix. shape: n x n
        self._W = tf.get_variable("W", shape=[self.embedding_size*2, self.embedding_size])

        # biasvector. shape: n
        self._b = tf.get_variable("b", shape=[self.embedding_size])

        # bilinear forms
        self._bilinear_forms = tf.get_variable("bilinear",
                shape=[self.embedding_size, self.embedding_size, self.embedding_size])

        # dropouts
        self._bilinear_forms = tf.layers.dropout(self._bilinear_forms,
                rate = self.dropout_bilinear_forms, training = self.is_training)
        self._W = tf.layers.dropout(self._W, rate = self.dropout_matrix,
                training = self.is_training)

        # get the composition
        self._architecture = self.compose(self._embeddings_u, self._embeddings_v, self._bilinear_forms,
                                          self._W, self._b)
        # add nonlinearity to the composition
        self._architecture = self.nonlinearity(self._architecture)

        # l2 normalize the composition
        self._architecture_normalized = super(
            BiLinear, self).l2_normalization_layer(
                tensor=self._architecture, axis=1)

    def compose(self, u, v, bilinear_forms, W, b):
        """
        composition of the form:
        p = g(uEv + W[u;v] + b)
        takes as input two tensors of any shape (tensors need to have at least rank 2)

        For a motivation of this model, see:
        
        - Recursive Deep Models for Semantic Compositionality Over a
          Sentiment Treebank, Socher et al., 2013
        - Reasoning With Neural Tensor Networks for Knowledge Base Completion,
          Socher et al, 2013
        """
        batch_size = tf.shape(u)[0]

        matrix = uv_affine_transform(u, v, W, b)

        v = tf.expand_dims(v, axis=2)

        # E' = u x E ->
        # (batch_size x embed_size) x (embed_size x embed_size x embed_size) ->
        # (batch_size x embed_size x embed_size)
        step1 = tf.tensordot(u, bilinear_forms, axes=1, name="bilinear_step1")

        # E' x v ->
        # (batch_size x embed_size x embed_size) x (batch_size x embed_size x 1) ->
        # (batch_size x embed_size x 1)
        bilinear = tf.matmul(step1, v, name="bilinear_step2")

        # (batch_size x embed_size x 1) -> (batch_size x embed_size)
        bilinear = tf.reshape(bilinear, [batch_size, self.embedding_size], name="bilinear_reshape")

        return bilinear + matrix

    @property
    def bilinear_forms(self):
        return self._bilinear_forms

    @property
    def dropout_bilinear_forms(self):
        return self._dropout_bilinear_forms

    @property
    def dropout_matrix(self):
        return self._dropout_matrix

    @property
    def nonlinearity(self):
        return self._nonlinearity

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

