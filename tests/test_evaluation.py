import unittest
from pathlib import Path

import numpy as np

import evaluation
import data

class EvaluationTest(unittest.TestCase):
    """
    This class tests the functionality of the evaluation.py script.
    This test suite can be ran with:
        python -m unittest -q tests.EvaluationTest
    """

    def setUp(self):
        test_data_dir = Path(__file__).resolve().parents[1]
        embeddings_file = str(test_data_dir.joinpath("test_data").joinpath("embeddings.txt"))
        self._predictions_file = str(test_data_dir.joinpath("test_data").joinpath("gold_standard.txt"))

        self._unknown_word_key = "<unk>"
        self._word_embeddings = data.read_word_embeddings(embeddings_file, self._unknown_word_key)

    def test_all_ranks(self):
        ranks = evaluation.get_all_ranks(predictions_file=self._predictions_file, word_embeddings=self._word_embeddings,
                                         max_rank=14, batch_size=3, path_to_ranks="")
        np.testing.assert_equal(ranks, np.full([8], 1))

    def test_n_nearest(self):
        ranks = evaluation.get_all_ranks(predictions_file=self._predictions_file, word_embeddings=self._word_embeddings,
                                         max_rank=5, batch_size=3, path_to_ranks="")
        np.testing.assert_equal(ranks, np.full([8], 1))

    def test_bad_prediction(self):
        compounds, prediction = evaluation.read_test_data(predictions_file=self._predictions_file, batch_size=3)
        prediction[2][1] = np.array([0.60971076, 0.792623999])
        ranks = evaluation.get_target_based_rank(composed_repr=prediction[2], targets=compounds[2], 
                        dictionary_embeddings=self._word_embeddings, max_rank=5)
        np.testing.assert_equal(ranks, [1, 5])

    def test_close_prediction(self):
        compounds, prediction = evaluation.read_test_data(predictions_file=self._predictions_file, batch_size=3)
        prediction[2][1] = np.array([0.18987054, 0.98180914])
        ranks = evaluation.get_target_based_rank(composed_repr=prediction[2], targets=compounds[2], 
                        dictionary_embeddings=self._word_embeddings, max_rank=5)
        np.testing.assert_equal(ranks, [1, 2])

    def test_ranks(self):
        predictions = np.array([[0.85202893, 0.52349469], [0.70422221, 0.70997963],
                                [0.67320558, 0.73945537], [0.67800539, 0.73505693]], dtype=np.float32)
        compounds = ["apfelbaum", "quarkstrudel", "kirschbaum", "kirschstrudel"]
        correct_ranks = [3, 1, 10, 8]
        predicted_ranks = evaluation.get_target_based_rank(composed_repr=predictions, targets=compounds, 
                                            dictionary_embeddings=self._word_embeddings, max_rank=14)
        np.testing.assert_equal(predicted_ranks, correct_ranks)

    def test_assertion(self):
        self.assertRaises(AssertionError,
                          lambda: evaluation.get_all_ranks(predictions_file=self._predictions_file,
                                                           word_embeddings=self._word_embeddings,
                                                           max_rank=20, batch_size=3, path_to_ranks=""))

    def test_quartiles_uneven(self):
        ranks = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
        quartiles, percent = evaluation.calculate_quartiles(ranks)
        result = [15, 40, 43]
        np.testing.assert_equal(quartiles, result)

    def test_quartiles_even(self):
        ranks = [7, 15, 36, 39, 40, 41]
        quartiles, percent = evaluation.calculate_quartiles(ranks)
        result = [15, 37.5, 40]
        np.testing.assert_equal(quartiles, result)
