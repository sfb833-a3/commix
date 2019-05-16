import tensorflow as tf

from models import TransWeight


class TransWeightFeatures(TransWeight):
    """
    Contains the architecture of a variation of the TransWeight composition model : The row-wise scalar variation.
    Takes as input two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size, nonlinearity, dropout_rate, transforms):
        super(TransWeightFeatures, self).__init__(embedding_size, nonlinearity, dropout_rate, transforms)

    def create_architecture(self):

        self._transformations_tensor = tf.get_variable("transformations_tensor",
                                                       shape=[self.transforms, 2 * self.embedding_size, self.embedding_size])
        self._transformations_bias = tf.get_variable("transformations_bias",
                                                     shape=[self.transforms, self.embedding_size])

        # transformation weight vector - weights the transformed representations in the previous step
        self._W = tf.get_variable("W", shape=[self.embedding_size])

        # bias vector for the combination tensor
        self._b = tf.get_variable("b", shape=[self.embedding_size])

        self._architecture = super(
            TransWeightFeatures, self).compose(
            u=self.embeddings_u,
            v=self.embeddings_v,
            transformations_tensor=self.transformations_tensor,
            transformations_bias=self.transformations_bias,
            W=self.W,
            b=self.b)

        self._architecture_normalized = super(
            TransWeightFeatures, self).l2_normalization_layer(self._architecture, 1)

    def weight(self, reg_uv, W, b):
        # transformations are weighted using W into a final composed representation
        weighted_uv = tf.reduce_sum(tf.multiply(reg_uv, W), axis=1)
        weighted_uv_bias = tf.add(weighted_uv, b)

        return weighted_uv_bias
