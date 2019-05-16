from abc import ABC, abstractmethod

import tensorflow as tf


class AbstractModel(ABC):
    """
    Defines the Abstract Model that all composition models should inherit from.
    Includes the properties all composition models should have and defines some general functions.
    """

    def __init__(self, embedding_size):
        self._is_trainable = True
        self._is_training = tf.placeholder_with_default(False, shape = ())
        self._embedding_size = embedding_size

        # placeholder to feed in the words and their embeddings
        self._lookup_init = tf.placeholder(tf.float32, shape=[None, self._embedding_size])
        
        self._lookup = tf.Variable(initial_value=self._lookup_init, validate_shape=False, trainable=False)

        self._u = tf.placeholder(tf.int64, shape=[None])
        self._v = tf.placeholder(tf.int64, shape=[None])

        # lookup of the embeddings
        self._embeddings_u = tf.nn.embedding_lookup(self._lookup, self._u)
        self._embeddings_v = tf.nn.embedding_lookup(self._lookup, self._v)

    @abstractmethod
    def compose(self, u, v):
        """All models should implement a composition function that takes two tensors (u and v) as input."""
        return

    @abstractmethod
    def create_architecture(self):
        """
        All models should implement a function that defines and creates the concrete architecture with nodes and
        operations with details depending on the concrete models.
        :return: the composition architecture
        """
        return

    @staticmethod
    def l2_normalization_layer(tensor, axis):
        """
        This method adds a normalization layer on top of a model were the input tensor
        is returned as L2 normalized tensor.
        :param tensor: tensor object that should be converted into normalized tensor
        :param axis: dimension along which normalization takes place
        :return: the normalized tensor object
        """
        return tf.nn.l2_normalize(tensor, axis, name='l2_normalize')


    @property
    def architecture(self):
        """This property stores the output of the graph."""
        return self._architecture

    @property
    def architecture_normalized(self):
        """This property stores the output of the graph in a normalized form."""
        return self._architecture_normalized

    @property
    def is_training(self):
        """This property stores the placeholder to indicate whether the current
        run is a training run. It is set to False by default."""
        return self._is_training

    @property
    def u(self):
        """This property represents one of the input tensors of the graph."""
        return self._u

    @property
    def embeddings_u(self):
        """This property represents the looked up embeddings for the first input of the graph."""
        return self._embeddings_u

    @property
    def v(self):
        """This property represents the other input tensor of the graph."""
        return self._v

    @property
    def embeddings_v(self):
        """This property represents the looked up embeddings for the second input of the graph."""
        return self._embeddings_v

    @property
    def is_trainable(self):
        """This property defines if the model is trainable or not"""
        return self._is_trainable

    @is_trainable.setter
    def is_trainable(self, val):
        self._is_trainable = val

    @property
    def lookup(self):
        """This property defines the embedding matrix of (pretrained) embeddings, the word vectors are looked up"""
        return self._lookup

    @property
    def lookup_init(self):
        """This property defines a placeholder for the lookup"""
        return self._lookup_init

    @property
    def embedding_size(self):
        """This property defines the embedding dimension"""
        return self._embedding_size

    def regularization(self):
        """This property defines the regularizer for a model. Per default this is just a constant of 0"""
        return tf.constant(0.0)