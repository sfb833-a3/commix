import unittest
from pathlib import Path

import numpy as np

import data
import evaluation
import evaluation_composed as evac


class EvaluationComposedBasedTest(unittest.TestCase):

    def setUp(self):
        test_data_dir = Path(__file__).resolve().parents[1]
        embeddings_file = str(test_data_dir.joinpath("test_data").joinpath("embeddings.txt"))
        self._predictions_file = str(test_data_dir.joinpath("test_data").joinpath("gold_standard.txt"))

        self._unknown_word_key = "<unk>"
        self._word_embeddings = data.read_word_embeddings(embeddings_file, self._unknown_word_key)

    def test_all_ranks(self):
        ranks = evac.get_all_ranks(predictions_file=self._predictions_file, word_embeddings=self._word_embeddings,
                                                  max_rank=14, batch_size=3, path_to_ranks="")
        np.testing.assert_equal(ranks, np.full([8], 1))

    def test_n_nearest(self):
        ranks = evac.get_all_ranks(predictions_file=self._predictions_file, word_embeddings=self._word_embeddings,
                                                  max_rank=5, batch_size=3, path_to_ranks="")
        np.testing.assert_equal(ranks, np.full([8], 1))

    def test_bad_prediction(self):
        compounds, prediction = evaluation.read_test_data(predictions_file=self._predictions_file, batch_size=3)
        prediction[2][1] = np.array([0.60971076, 0.792623999])
        ranks = evac.get_composed_based_rank(composed_repr=prediction[2], targets=compounds[2],
                                                            dictionary_embeddings=self._word_embeddings, max_rank=5)
        np.testing.assert_equal(ranks, [1, 5])

    def test_close_prediction(self):
        compounds, prediction = evaluation.read_test_data(predictions_file=self._predictions_file, batch_size=3)
        prediction[2][1] = np.array([0.18987054, 0.98180914])
        ranks = evac.get_composed_based_rank(composed_repr=prediction[2], targets=compounds[2],
                                                            dictionary_embeddings=self._word_embeddings, max_rank=5)
        np.testing.assert_equal(ranks, [1, 2])

    def test_ranks(self):
        predictions = np.array([[0.85202893, 0.52349469], [0.70422221, 0.70997963],
                                [0.67320558, 0.73945537], [0.67800539, 0.73505693]], dtype=np.float32)
        compounds = ["apfelbaum", "quarkstrudel", "kirschbaum", "kirschstrudel"]
        correct_ranks = [3, 1, 9, 14]
        predicted_ranks = evac.get_composed_based_rank(composed_repr=predictions, targets=compounds,
                                                                      dictionary_embeddings=self._word_embeddings, max_rank=14)
        np.testing.assert_equal(predicted_ranks, correct_ranks)

    def test_assertion(self):
        self.assertRaises(AssertionError,
                          lambda: evaluation.get_all_ranks(predictions_file=self._predictions_file,
                                                           word_embeddings=self._word_embeddings,
                                                           max_rank=20, batch_size=3, path_to_ranks=""))
