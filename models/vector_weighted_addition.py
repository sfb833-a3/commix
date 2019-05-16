import tensorflow as tf

from models import AbstractModel


class VectorWeightedAddition(AbstractModel):
    """
    Contains the architecture of the vector weighted addition model. Takes as input of two vectors (or batches
    of vectors). Its superclass is the AbstractModel, all properties are defined and described there.
    """

    def __init__(self, embedding_size):
        super(VectorWeightedAddition, self).__init__(embedding_size)

    def create_architecture(self):
        """
        Defines the architecture of the vector weighted addition model. The input is defined by the placeholders
        u and v. These are matrices containing the indices of the wordembeddings for the heads and the modifier.
        The resulting composition is stored in self._architecture. A L2 normalized output can be obtained by via
        self._normalized_architecture. The vectors are randomly initialized and can be optimized during training.
        :param batch_size: can be None or a concrete integer that defines the number of instances in one batch
        :param lookup: a lookup table that contains word indices and corresponding word embeddings
        """
        self._a = tf.get_variable("a", shape=[self.embedding_size])
        self._b = tf.get_variable("b", shape=[self.embedding_size])

        self._architecture = self.compose(
            self.embeddings_u, self.embeddings_v, self._a, self._b)

        self._architecture_normalized = super(
            VectorWeightedAddition,
            self).l2_normalization_layer(
            self._architecture,
            1)

    def compose(self, u, v, a, b):
        """
        composition of the form: p = a * u + b * v
        :param u: a tensor object
        :param v: a tensor object with the same shape as u
        :param a: a tensor (vector)
        :param b: a tensor (vector)
        :return:
        """
        return tf.add((tf.multiply(a, u)), (tf.multiply(b, v)))


    @property
    def a(self):
        """This property stores the vector a that is needed for the vector weighted composition.
        It is randomly initialized."""
        return self._a

    @property
    def b(self):
        """This property stores the vector b that is needed for the vector weighted composition.
        It is randomly initialized."""
        return self._b
