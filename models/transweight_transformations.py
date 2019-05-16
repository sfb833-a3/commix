import tensorflow as tf

from models import TransWeight


class TransWeightTransformations(TransWeight):
    """
    Contains the architecture of a variation of the TransWeight composition model : the column-wise scalar variation.
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size, nonlinearity, dropout_rate, transforms):
        super(TransWeightTransformations, self).__init__(embedding_size, nonlinearity, dropout_rate, transforms)

    def create_architecture(self):

        self._transformations_tensor = tf.get_variable("transformations_tensor",
                                                       shape=[self.transforms, 2 * self.embedding_size, self.embedding_size])
        self._transformations_bias = tf.get_variable("transformations_bias",
                                                     shape=[self.transforms, self.embedding_size])

        # weight vector of dimension t - weights every transformation separately
        self._W = tf.get_variable("W", shape=[self.transforms, 1])

        # bias vector for the combination tensor
        self._b = tf.get_variable("b", shape=[self.embedding_size])

        self._architecture = super(
            TransWeightTransformations, self).compose(
            u=self.embeddings_u,
            v=self.embeddings_v,
            transformations_tensor=self.transformations_tensor,
            transformations_bias=self.transformations_bias,
            W=self.W,
            b=self.b)

        self._architecture_normalized = super(
            TransWeightTransformations, self).l2_normalization_layer(self._architecture, 1)

    def weight(self, reg_uv, W, b):
        # transformations are weighted using W into a final composed representation
        weighted_uv = tf.multiply(W, reg_uv)
        weighted_uv = tf.reduce_sum(weighted_uv, axis=1)
        weighted_uv_bias = tf.add(weighted_uv, b)

        return weighted_uv_bias
