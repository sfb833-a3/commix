import tensorflow as tf

from models import AbstractModel


class ScalarWeightedAddition(AbstractModel):
    """
    Contains the architecture of the weighted addition model (scalar).
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size):
        super(ScalarWeightedAddition, self).__init__(embedding_size)

    def create_architecture(self):
        """
        Defines the architecture of the scalar weighted addition model. The input is defined by the placeholders
        u and v. These are matrices containing the indices of the wordembeddings for the heads and the modifier.
        The resulting composition is stored in self._architecture. A L2 normalized output can be obtained via
        self._normalized_architecture. The scalars are randomly initialized and can be optimized during training.
        :param batch_size: can be None or a concrete integer that defines the number of instances in one batch
        :param lookup: a lookup table that contains word indices and corresponding word embeddings
        """
        self._alpha = tf.get_variable("alpha", shape=())
        self._beta = tf.get_variable("beta", shape=())

        self._architecture = self.compose(self.embeddings_u, self.embeddings_v, self.alpha, self.beta)

        self._architecture_normalized = super(ScalarWeightedAddition, self).l2_normalization_layer(self._architecture, 1)

    def compose(self, u, v, alpha, beta):
        """
        composition of the form: p = alpha * u + beta * v
        Tensors u and v can have any shape but their shapes must be equal.
        :param u: a tensor object
        :param v: a tensor object having the same shape as u
        :param alpha: a scalar that u can be multiplied with
        :param beta: a scalar that v can be multiplied with
        :return: a tensor object that is the result of the applied composition function
        """
        return tf.add((tf.scalar_mul(alpha, u)),(tf.scalar_mul(beta, v)))

    @property
    def alpha(self):
        """This property stores the scalar alpha needed for the composition function"""
        return self._alpha

    @property
    def beta(self):
        """This property stores the scalar beta needed for the composition function"""
        return self._beta