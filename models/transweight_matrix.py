import tensorflow as tf

from models import TransWeight


class TransWeightMatrix(TransWeight):
    """
    Contains the architecture of a variation of the TransWeight composition model: the transweight vector variation.
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size, nonlinearity, dropout_rate, transforms):
        super(TransWeightMatrix, self).__init__(embedding_size, nonlinearity, dropout_rate, transforms)


    def create_architecture(self):

        self._transformations_tensor = tf.get_variable("transformations_tensor",
                                                       shape=[self.transforms, 2 * self.embedding_size, self.embedding_size])
        self._transformations_bias = tf.get_variable("transformations_bias",
                                                     shape=[self.transforms, self.embedding_size])

        # rank 3 combination tensor of shape 1 x t x n - combines the transformed representations in the previous step
        self._W = tf.get_variable("W", shape=[1, self.transforms, self.embedding_size])

        # bias vector for the combination tensor
        self._b = tf.get_variable("b", shape=[self.embedding_size])

        self._architecture = super(
            TransWeightMatrix, self).compose(
            u=self.embeddings_u,
            v=self.embeddings_v,
            transformations_tensor=self.transformations_tensor,
            transformations_bias=self.transformations_bias,
            W=self.W,
            b=self.b)

        self._architecture_normalized = super(
            TransWeight, self).l2_normalization_layer(self._architecture, 1)

    def weight(self, reg_uv, W, b):
        # transformations are weighted using W into a final composed representation
        weighted_uv = tf.reduce_sum(tf.multiply(reg_uv, W), axis=1)
        weighted_uv_bias = tf.add(weighted_uv, b)

        return weighted_uv_bias
