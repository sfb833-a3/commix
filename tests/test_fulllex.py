import numpy as np
import tensorflow as tf

from tests import TestBase
from models import FullLex, Regularizer
from utils import matrix_mapping
from utils.ops import init_hashtable
from training_graph import TrainingGraph, RunMode


class FullLexTest(TestBase):
    """
    This class tests the functionality of the FullLex model.
    This test suite can be ran with:
        python -m unittest -q tests.FullLexTest
    """
    def setUp(self):
        """
        This method calls the setUp of the superclass and defines the specific composition model as a property of this
        testclass. The composition model of this test class is the FullLex model.
        """
        super(FullLexTest, self).setUp()
        self._mh_map, self._unk_matrix_idx = matrix_mapping.create_matrix_mapping(self._db.mh_set, self._db.unk_vector_id)
        self._mh_map_vd = matrix_mapping.create_matrix_mapping_with_neighbours(self._vd.mh_set,
                                                                               self._embedding_model, self._mh_map)
        self._comp_model = FullLex(embedding_size=self._embedding_dim, mh_index_map=self._mh_map,
                unk_matrix_id=self._unk_matrix_idx,
                nonlinearity=tf.identity,
                dropout_rate=0.0,
                regularizer=Regularizer.l1_regularizer)

    def test_composition(self):
        """
        Tests if the composition method itself is correct. The composition method is
        p = W[Vu;Uv] + b

        """

        u = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        v = np.array([[2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        U = np.full(shape=(3, 2, 2), fill_value=2.0)
        V = np.full(shape=(3, 2, 2), fill_value=1.0)
        W = np.full(shape=(4, 2), fill_value=2.0)
        W[1][0] = 1
        W[2][0] = 3
        W[3][0] = 5
        b = np.array([3.0, 3.0])
        result = [[57.0, 35.0], [79.0, 51.0], [60.0, 39.0]]
        with tf.Session() as sess:
            comp = sess.run(self._comp_model.compose(u, v, U, V, W, b))
        np.testing.assert_allclose(comp, result)

    def testL1RegularizationMethod(self):
        """Test if the L1 regularization method of the FullLex model returns the correct result"""
        matrix_U = np.array([
            [[2, 3], [4, 3]],
            [[5, 2], [1, 2]],
            [[1, 2], [2, 2]]], dtype=np.float32)

        matrix_V = np.array([
            [[3, 3], [2, 1]],
            [[1, 2], [3, 4]],
            [[4, 5], [5, 1]]], dtype=np.float32)
        alpha = 0.1
        cosine_loss = 0.5
        result = 3.5
        with tf.Session() as sess:
            reg_term = sess.run(self._comp_model.l1_regularizer(matrix_U=matrix_U, matrix_V=matrix_V))
        reg_loss = alpha * reg_term + cosine_loss
        np.testing.assert_equal(result, reg_loss)

    def testDotRegularizationMethod(self):
        """Tests if the dot regularization method of the FullLex model returns the correct result"""
        u = np.array([[0.91822219, 0.39606566], [0.62372849, 0.78164107]])
        v = np.array([[0.76826709, 0.64012942], [0.77653667, 0.63007206]])

        Uv = np.array([[0.76836709, 0.64012942], [1.794877065, 1.7216447600000002]])
        Vu = u
        with tf.Session() as sess:
            regularized_value = sess.run(self._comp_model.dot_regularizer(Vu=Vu, u=u, Uv=Uv, v=v))
        np.testing.assert_almost_equal(regularized_value, 0.0034367, decimal=4)

    def testL1Regularization(self):
        """Test if the Fulllex Composition model can be run with regularization
         and returns a different regularized loss than the unregularized one"""


        training_graph = super(FullLexTest, self).create_training_graph(comp_model=self._comp_model,
                                                                        learning_rate=0.01,
                                                                        regularization=0.5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_graph.model.lookup_init: self._lookup})
            for epoch in range(3):
                for tidx in range(self._db.no_batches):
                    loss, reg_loss, _ = sess.run(
                        [training_graph.loss, training_graph.reg_loss, training_graph.train_op],
                        feed_dict={training_graph.original_vector: self._db.compound_batches[tidx],
                                   training_graph.model._u: self._db.modifier_batches[tidx],
                                   training_graph.model._v: self._db.head_batches[tidx]})
                    np.testing.assert_equal(loss != reg_loss, True)

    def testDotRegularization(self):
        """Test if the FullLex Composition model has a dot_regularization value of 0 (so loss = regularized loss)
        for the first batch, first run. To test this regularizer a new composition model has to be
        initialized with the dot regularizer."""
        tf.reset_default_graph()
        fulllex_model = FullLex(embedding_size=self._embedding_dim, mh_index_map=self._mh_map,
                unk_matrix_id=self._unk_matrix_idx,
                nonlinearity=tf.identity,
                dropout_rate=0.0,
                regularizer=Regularizer.dot_regularizer)

        training_graph = super(FullLexTest, self).create_training_graph(comp_model=fulllex_model,
                                                                        learning_rate=0.01,
                                                                        regularization=0.5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_graph.model.lookup_init: self._lookup})
            loss, reg_loss, _ = sess.run(
                [training_graph.loss, training_graph.reg_loss, training_graph.train_op],
                feed_dict={training_graph.original_vector: self._db.compound_batches[0],
                           training_graph.model._u: self._db.modifier_batches[0],
                           training_graph.model._v: self._db.head_batches[0]})
            np.testing.assert_almost_equal(loss, reg_loss, decimal=5)

    def test_matrix_mapping_unknown_word(self):
        """
        After training matrices for all words in training data, when an unknown word is encountered in the validation
        data (Zitrone), it should be mapped to the unknown matrix.
        """
        np.random.seed(1)
        sess = tf.Session()
        init_hashtable(mh_map=self._mh_map, index_hash=self._comp_model.index_hash, sess=sess)
        training_graph = super(FullLexTest, self).create_training_graph(comp_model=self._comp_model,
                                                                        learning_rate=0.3,
                                                                        regularization=0.0)
        with sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_graph.model.lookup_init: self._lookup})
            for epoch in range(10):
                for tidx in range(self._db.no_batches):
                    loss = sess.run([training_graph.loss],
                                    feed_dict={training_graph.original_vector: self._db.compound_batches[tidx],
                                               training_graph.model._u: self._db.modifier_batches[tidx],
                                               training_graph.model._v: self._db.head_batches[tidx]})

            # for all words in training data, matrices were trained.
            # run the lookup for words in the validation data and lookup the matrix for anything unknown. The matrix for
            # an unknown word and the unknown matrix should be equal.
            word_matrices, lookup_table = sess.run([training_graph.model.matrix_U, training_graph.model.matrix_lookup],
                                                   feed_dict={
                                                       training_graph.original_vector: self._vd.compound_batches[0],
                                                       training_graph.model._u: self._vd.modifier_batches[0],
                                                       training_graph.model._v: self._vd.head_batches[0]})
            # lookup the 'unknown matrix'
            unk_matrix = lookup_table[self._unk_matrix_idx]
            # the word Zitrone is an unknown word, it should have the unknown matrix as word matrix
            zitrone_matrix = word_matrices[0]
        np.testing.assert_allclose(unk_matrix, zitrone_matrix)

    def test_matrix_mapping_unseen_word(self):
        """
        Tests the training and mapping of the word specific matrices. For all words in training data, specific matrices
        are trained. When encountering a word in the validation data, that has not been in training, a word vector
        for this word is present, but there won't be a word-specific matrix. Thus during validation or prediction the
        matrix of the unknown word will be used for composition.
        """
        np.random.seed(1)
        sess = tf.Session()
        init_hashtable(mh_map=self._mh_map, index_hash=self._comp_model.index_hash, sess=sess)
        training_graph = super(FullLexTest, self).create_training_graph(comp_model=self._comp_model,
                                                                        learning_rate=0.3,
                                                                        regularization=0.0)
        with sess:
            sess.run(tf.global_variables_initializer(), feed_dict={training_graph.model.lookup_init: self._lookup})
            for epoch in range(10):
                for tidx in range(self._db.no_batches):
                    loss = sess.run([training_graph.loss],
                        feed_dict={training_graph.original_vector: self._db.compound_batches[tidx],
                                   training_graph.model._u: self._db.modifier_batches[tidx],
                                   training_graph.model._v: self._db.head_batches[tidx]})

            # for all words in training data, matrices were trained.
            # run the lookup for words in the validation data and lookup the matrix for anything unknown. The matrix for
            # words not seen in training and the unknown matrix should be equal.
            word_matrices, lookup_table = sess.run([training_graph.model.matrix_U, training_graph.model.matrix_lookup],
                feed_dict={training_graph.original_vector: self._vd.compound_batches[1],
                           training_graph.model._u: self._vd.modifier_batches[1],
                           training_graph.model._v: self._vd.head_batches[1]})
            # lookup the 'unknown matrix'
            unk_matrix = lookup_table[self._unk_matrix_idx]
            # the words Leder and Stamm are not in the training data, they should have the unknown matrix as word matrix
            leder_matrix = word_matrices[0]
            stamm_matrix = word_matrices[1]
        np.testing.assert_allclose(unk_matrix, leder_matrix)
        np.testing.assert_allclose(unk_matrix, stamm_matrix)

    def test_matrix_mapping_nearest_neightbour(self):
        """
        Tests whether a word that has not been seen in training is mapped to the matrix of it's nearest neighbour
        """
        np.random.seed(1)
        sess = tf.Session()
        init_hashtable(mh_map=self._mh_map, index_hash=self._comp_model.index_hash, sess=sess)
        training_graph = super(FullLexTest, self).create_training_graph(comp_model=self._comp_model,
                                                                        learning_rate=0.3,
                                                                        regularization=0.0)
        # matrices are trained for all words in training
        with sess:
            saver = tf.train.Saver(max_to_keep=0)
            sess.run(tf.global_variables_initializer(), feed_dict={training_graph.model.lookup_init: self._lookup})
            for epoch in range(10):
                for tidx in range(self._db.no_batches):
                    loss = sess.run([training_graph.loss],
                                    feed_dict={training_graph.original_vector: self._db.compound_batches[tidx],
                                               training_graph.model._u: self._db.modifier_batches[tidx],
                                               training_graph.model._v: self._db.head_batches[tidx]})
            saver.save(sess, str(self._test_data_dir.joinpath("test_data").joinpath("example_model")))
        # during validation the model encounters words not seen in training (Leder (idx:16)) The nearest neighbour of
        # Leder ist Apfel. So the wordmatrix used for Leder should be equal to the wordmatrix used for Apfel (idx:0)
        with tf.Session() as sess:
            with tf.variable_scope("model", reuse=True):
                best_model = TrainingGraph(composition_model=self._comp_model,
                                           batch_size=None,
                                           learning_rate=0.3,
                                           run_mode=RunMode.validation,
                                           alpha=0.0)
                if saver != None:
                    saver.restore(sess, str(self._test_data_dir.joinpath("test_data").joinpath("example_model")))
                init_hashtable(mh_map=self._mh_map_vd, index_hash=self._comp_model.index_hash, sess=sess)
                results = []
                for vidx in range(self._vd.no_batches):
                    result = sess.run(
                        [best_model.model.matrix_U],
                        feed_dict={best_model.original_vector: self._vd.compound_batches[vidx],
                                   best_model.model._u: self._vd.modifier_batches[vidx],
                                   best_model.model._v: self._vd.head_batches[vidx]})
                    results.extend(result)
        matrix_apfel = results[0][1]
        matrix_leder = results[1][0]
        # index 16 should be mapped to the matrix of its closest index (which is index 0) so the first and the last
        # matrix of the batch should be the same, as we are looking up [0 2 16], because the mapping was updated
        np.testing.assert_allclose(matrix_apfel, matrix_leder)

    def test_matrix_mapping(self):
        """Tests if the vector_idx-matrix-idx mapping is correct for train and test data"""
        # nearest neighbour leder (idx 16): apfel (idx 0)
        # nearest neighbour stamm (idx 17): baum (idx 1)
        # unknown words (idx 6) is mapped to unknown matrix (idx 6)
        train_mh_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 11: 7}
        valid_mh_map = {0: 0, 1: 1, 2: 2, 5: 5, 6: 6, 11: 7, 16: 0, 17: 1}
        np.testing.assert_equal(train_mh_map, self._mh_map)
        np.testing.assert_equal(valid_mh_map, self._mh_map_vd)

