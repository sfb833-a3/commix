from types import SimpleNamespace

import numpy as np

from utils.gensim_utils import read_gensim_model

def add_unknown_embedding(embedding_model, unknown_word_key):
    indices = np.random.randint(0, embedding_model.wv.syn0.shape[0], 1000)
    unknown_embedding = np.mean(embedding_model.wv.syn0[indices], axis=0)
    embedding_model.wv.add(unknown_word_key, unknown_embedding, False)
    return embedding_model

def read_word_embeddings(embeddings_file, unknown_word_key):
    """Reads pretrained word embeddings from a file. Expects gensim format.
    Vectors are normalized using the L2 norm. If the specified unknown word key is
    not found in the vocabulary, an unknown word embedding is created and added to
    the vector space."""

    model = read_gensim_model(embeddings_file)
    
    if unknown_word_key not in model.wv.vocab:
        unk_model = add_unknown_embedding(model, unknown_word_key)
        model = unk_model
    
    model.wv.syn0 /= np.expand_dims(np.linalg.norm(model.wv.syn0, axis=-1), axis=-1)

    return model

def generate_instances(batch_size, file_path, word_index, separator, unknown_word_key):
    """
    Reads composition dataset files (txt format), with 3 entries on each line, 
    e.g. Apfel Baum Apfelbaum.

    Each entry is converted into word indices that can be looked with a lookup table.
    Converts data into batches. All batches are saved to lists.
    """

    modifier_batches, head_batches, compound_batches = [], [], []
    modifier_vec, head_vec, compound_vec = [], [], []
    text_compounds = []

    batch_index = 0
    total_size = 0

    unk_id = get_word_id(unknown_word_key, word_index, unknown_word_key)
    mh_set = set()
    mh_set.add(unk_id)

    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            line_parts = line.strip().split(separator)
            assert (len(line_parts) == 3), "error: wrong number of elements on line"

            modifier = line_parts[0]
            head = line_parts[1]
            compound = line_parts[2]

            text_compounds.append(compound)

            modifier_id = get_word_id(modifier, word_index, unknown_word_key)
            head_id = get_word_id(head, word_index, unknown_word_key)
            compound_id = get_word_id(compound, word_index, unknown_word_key)

            mh_set.add(modifier_id)
            mh_set.add(head_id)

            modifier_vec.append(modifier_id)
            head_vec.append(head_id)
            compound_vec.append(compound_id)

            batch_index += 1
            total_size += 1

            if batch_index == batch_size:
                modifier_batches.append(np.asarray(modifier_vec, dtype=np.int64))
                head_batches.append(np.asarray(head_vec, dtype=np.int64))
                compound_batches.append(np.asarray(compound_vec, dtype=np.int64))

                batch_index = 0
                modifier_vec, head_vec, compound_vec = [], [], []

    # create a new batch only if there is more data
    if batch_index > 0:
        modifier_batches.append(np.asarray(modifier_vec, dtype=np.int64))
        head_batches.append(np.asarray(head_vec, dtype=np.int64))
        compound_batches.append(np.asarray(compound_vec, dtype=np.int64))

    assert(len(modifier_batches) == len(head_batches) == len(compound_batches)), "error: inconsistent batch size"
    assert(total_size == sum([len(batch) for batch in modifier_batches])), "error: batches missing data"

    data_batches = SimpleNamespace(
        modifier_batches = modifier_batches,
        head_batches = head_batches,
        compound_batches = compound_batches,
        text_compounds = text_compounds,
        no_batches = len(compound_batches),
        total_size = total_size,
        mh_set = mh_set,
        unk_vector_id = unk_id
    )

    print("%d unique modifiers and heads in the dataset, including unknown vector(s)" % len(mh_set))
    return data_batches

def get_word_id(word, word_index, unknown_word_key):
    """Retrieves the index of a word if the word is part of the word_index.
    Otherwise returns the unknown word index."""

    if (word in word_index):
        idx = word_index[word].index
    else:
        idx = word_index[unknown_word_key].index
    return idx
