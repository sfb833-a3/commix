import unittest

from pathlib import Path

import data

class DataTest(unittest.TestCase):
    """
    This class tests the functionality of the data.py script.
    This test suite can be ran with:
        python -m unittest -q tests.DataTest
    """
    
    def setUp(self):
        test_data_dir = Path(__file__).resolve().parents[1]
        embeddings_file = str(test_data_dir.joinpath("test_data").joinpath("embeddings.txt"))
        self._train_dataset = str(test_data_dir.joinpath("test_data").joinpath("train_data.txt"))
        self._validation_dataset = str(test_data_dir.joinpath("test_data").joinpath("valid_data.txt"))
        
        self._unknown_word_key = "<unk>"
        self._separator = " "
        gensim_model = data.read_word_embeddings(embeddings_file, self._unknown_word_key)
        self._word_index = gensim_model.wv.vocab

    def test_equal_length(self):
        """all batch lists should have equal length"""

        batch_size = 2
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        self.assertTrue(len(self._db.modifier_batches) == len(self._db.head_batches) 
            == len(self._db.compound_batches))

    def test_batch_shape(self):
        """batches should have equal shapes"""

        batch_size = 2
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        for i in range(self._db.no_batches):
            self.assertTrue(self._db.modifier_batches[i].shape == self._db.head_batches[i].shape 
                == self._db.compound_batches[i].shape)

    def test_small_last_batch(self):
        """the last batch should contain the remainder of data if the 
        dataset size is not a multiple of batch size"""

        batch_size = 4
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        self.assertEqual(self._db.no_batches, 2)
        self.assertEqual(self._db.modifier_batches[0].shape, (4,))
        self.assertEqual(self._db.modifier_batches[1].shape, (2,))

    def test_exact_multiple_batch(self):
        """if the dataset size is an exact multiple of the batch size,
        expect dataset size/batch size batches"""

        batch_size = 2
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        self.assertEqual(self._db.no_batches, 3)
        self.assertEqual(self._db.modifier_batches[0].shape, (2,))
        self.assertEqual(self._db.modifier_batches[1].shape, (2,))
        self.assertEqual(self._db.modifier_batches[2].shape, (2,))

    def test_one_big_batch(self):
        """ if the dataset size equals the batch size, 
        one batch of size batch_size should be returned"""

        batch_size = 6
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        self.assertEqual(self._db.no_batches, 1)
        self.assertEqual(self._db.modifier_batches[0].shape, (6,))

    def test_dataset_size(self):
        """ensure all expected data is loaded"""

        batch_size = 2
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        self.assertEqual(self._db.total_size, 6)        

    def test_unknown_word_exists(self):
        """ the unknown vector should be loaded from the word embeddings; <unk> has index 6"""

        batch_size = 2
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        unknown_id = data.get_word_id("supercalifragilisticexpialidocious", 
            self._word_index, self._unknown_word_key)
        self.assertEqual(unknown_id, 6)

    def test_unknown_word_idx(self):
        """ the unknown index should be return for unknown words"""

        batch_size = 6
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._validation_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        unk_index = self._word_index[self._unknown_word_key].index
        self.assertEqual(self._db.modifier_batches[0][0], unk_index) # zitrone, unk modifier

    def test_word_idx(self):
        """ ensure word indices are correctly returned"""

        batch_size = 6
        self._db = data.generate_instances(batch_size=batch_size, file_path=self._train_dataset,
                                           word_index=self._word_index,
                                           unknown_word_key=self._unknown_word_key, 
                                           separator=self._separator)

        self.assertEqual(self._db.modifier_batches[0][1], 2) # quark
        self.assertEqual(self._db.head_batches[0][1], 11) # strudel
        self.assertEqual(self._db.compound_batches[0][1], 10) # quarkstrudel
