import tensorflow as tf

from models import AbstractModel


class Addition(AbstractModel):
    """This class defines the simple addition model"""

    def __init__(self, embedding_size):
        super(Addition, self).__init__(embedding_size)

        # The Addition is not trainable
        self._is_trainable = False

    def create_architecture(self):

        p = self.compose(self.embeddings_u, self.embeddings_v)
        self._architecture = p

        self._architecture_normalized = super(
            Addition, self).l2_normalization_layer(self._architecture, 1)

    def compose(self, u, v):
        """simple composition function: p = u + v"""
        return tf.add(u, v)
